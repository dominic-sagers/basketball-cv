# Basketball-CV Android Frontend Specification

## 1. Overview

This document specifies the design and implementation of an Android video recording and streaming application that serves as the frontend for the basketball-cv real-time tracking system. The app captures video from the device camera and uploads **chunked video frames** to the main processing pipeline with strict correctness guarantees.

### Key Principles
- **Correctness over speed**: Prioritize frame integrity over throughput
- **Sequential chunk validation**: Pipeline only processes a chunk after verifying the previous chunk has no dropped frames
- **Bounded upload backpressure**: App only uploads the next chunk after receiving explicit confirmation from the pipeline
- **Graceful degradation**: Network failures do not result in silent frame loss

---

## 2. Architecture

### 2.1 High-Level Flow

```
Android App                          Main Pipeline
─────────────────────────────────────────────────────
┌─────────────┐
│Record Video │ → Chunk 0 (frames 0-29)
│   Loop      │
└──────┬──────┘
       │         [UPLOAD with frame metadata]
       │         ──────────────────────────────────→
       │                                    ┌──────────────────┐
       │                                    │Chunk Healthcheck │
       │                                    │- Verify count    │
       │                                    │- Check CRC/hash  │
       │                                    │- Detect gaps     │
       │                                    └────────┬─────────┘
       │                                             │
       │                    [ACK: OK or RESEND]     │
       │         ←──────────────────────────────────┘
       │
       ├─ Wait for ACK
       │
       ├─ ACK OK?
       │  ├─YES: Start Chunk 1
       │  └─NO: Retry Chunk 0
       │
       ▼
Chunk 1 (frames 30-59) [repeat]
```

### 2.2 Component Breakdown

#### Android App
- **CameraManager**: Captures frames at fixed FPS (e.g., 30 FPS)
- **ChunkWriter**: Buffers frames into fixed-size chunks
- **FrameMetadata**: Tracks frame timestamps, sequence numbers, frame sizes
- **ChunkUploader**: Handles HTTP/multipart uploads with retry logic
- **StateManager**: Tracks recording state, upload state, buffered chunks
- **NetworkMonitor**: Detects connectivity changes, throttles uploads

#### Pipeline (Main Machine)
- **ChunkReceiver**: HTTP endpoint that receives multipart uploads
- **ChunkHealthcheck**: Validates chunk integrity before queuing for processing
- **ChunkProcessor**: Processes validated chunks (existing YOLO pipeline)
- **UploadAckServer**: Sends confirmation messages to app

---

## 3. Chunk Format & Protocol

### 3.1 Chunk Structure

Each chunk contains:
- **Chunk Header** (JSON metadata)
- **Frame Data** (raw H.264/VP9 frames or RAW frames)
- **Frame Index** (array of offsets and sizes)
- **Chunk Checksum** (CRC32 or SHA256)

```json
{
  "chunk_id": "camera_0_chunk_1234",
  "camera_id": "camera_0",
  "timestamp_start_ms": 1713700123456,
  "timestamp_end_ms": 1713700124000,
  "frame_count": 30,
  "expected_frame_count": 30,
  "fps": 30,
  "resolution": "1920x1080",
  "encoding": "h264",
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 1713700123456,
      "sequence_number": 5430,
      "size_bytes": 45230,
      "offset_bytes": 0,
      "keyframe": true
    },
    ...
  ],
  "checksum_algorithm": "crc32",
  "checksum_value": "a1b2c3d4"
}
```

### 3.2 Upload Protocol

**Method**: `POST /api/v1/chunks/upload`

**Request**:
```
Content-Type: multipart/form-data

Field: metadata (JSON) → chunk header
Field: video (binary) → concatenated frame data
Field: checksum (string) → validation hash
```

**Response (202 Accepted)**:
```json
{
  "status": "received",
  "chunk_id": "camera_0_chunk_1234",
  "timestamp_received_ms": 1713700124500,
  "message": "Chunk queued for healthcheck"
}
```

### 3.3 Healthcheck Protocol

**Method**: `GET /api/v1/chunks/{chunk_id}/status`

**Response (200 OK — Passed)**:
```json
{
  "chunk_id": "camera_0_chunk_1234",
  "status": "healthy",
  "validation": {
    "frame_count_match": true,
    "checksum_valid": true,
    "no_sequence_gaps": true,
    "no_timestamp_gaps": true,
    "frames_in_order": true
  },
  "message": "Chunk passed healthcheck. Safe to proceed.",
  "processing_started_ms": 1713700125000
}
```

**Response (400 Bad Request — Failed)**:
```json
{
  "chunk_id": "camera_0_chunk_1234",
  "status": "unhealthy",
  "validation": {
    "frame_count_match": false,
    "expected": 30,
    "received": 28,
    "missing_sequences": [10, 23]
  },
  "message": "Chunk has 2 dropped frames. Please resend.",
  "retry_deadline_ms": 1713700130000
}
```

---

## 4. Android App Design

### 4.1 Core Modules

#### CameraManager
```kotlin
class CameraManager(
    val targetFps: Int = 30,
    val resolution: Size = Size(1920, 1080),
    val encoding: String = "h264"  // or "vp9", "raw"
) {
    fun startRecording(onFrame: (Frame) -> Unit)
    fun stopRecording()
    fun getCurrentStats(): CameraStats  // FPS, dropped frames, etc.
}

data class Frame(
    val sequenceNumber: Long,
    val timestampMs: Long,
    val data: ByteArray,
    val size: Int,
    val isKeyframe: Boolean
)

data class CameraStats(
    val capturedFrames: Long,
    val droppedFrames: Long,
    val averageFps: Double
)
```

#### ChunkWriter
```kotlin
class ChunkWriter(
    val chunkSizeFrames: Int = 30,  // 1 second at 30 FPS
    val cameraId: String = "camera_0"
) {
    fun addFrame(frame: Frame): Chunk?  // Returns completed chunk or null
    fun flush(): Chunk?  // Force finalize current chunk
    fun getStats(): WriterStats
}

data class Chunk(
    val chunkId: String,
    val cameraId: String,
    val frames: List<Frame>,
    val timestampStartMs: Long,
    val timestampEndMs: Long,
    val checksum: String
)

data class WriterStats(
    val totalChunksCreated: Long,
    val framesInCurrentChunk: Int
)
```

#### ChunkUploader
```kotlin
class ChunkUploader(
    val backendUrl: String,
    val timeoutMs: Long = 30000,
    val maxRetries: Int = 5
) {
    suspend fun uploadChunk(chunk: Chunk): UploadResult
    fun cancelPendingUploads()
    fun getStats(): UploaderStats
}

sealed class UploadResult {
    data class Success(val chunkId: String, val timestampReceivedMs: Long) : UploadResult()
    data class Failure(val chunkId: String, val reason: String, val retryable: Boolean) : UploadResult()
    data class NetworkError(val error: Exception) : UploadResult()
}

data class UploaderStats(
    val totalChunksUploaded: Long,
    val totalRetries: Long,
    val failedChunks: Long,
    val averageUploadTimeMs: Long
)
```

#### HealthcheckPoller
```kotlin
class HealthcheckPoller(
    val backendUrl: String,
    val pollIntervalMs: Long = 1000,
    val timeoutMs: Long = 60000  // Max wait for healthcheck result
) {
    suspend fun waitForHealthcheck(chunkId: String): HealthcheckResult
    fun cancelPending()
}

sealed class HealthcheckResult {
    data class Healthy(val chunkId: String, val processingStartedMs: Long) : HealthcheckResult()
    data class Unhealthy(
        val chunkId: String,
        val droppedFrameIndices: List<Int>,
        val retryDeadlineMs: Long
    ) : HealthcheckResult()
    data class Timeout(val chunkId: String) : HealthcheckResult()
    data class NetworkError(val error: Exception) : HealthcheckResult()
}
```

#### RecordingOrchestrator (State Machine)
```kotlin
class RecordingOrchestrator(
    val camera: CameraManager,
    val chunkWriter: ChunkWriter,
    val uploader: ChunkUploader,
    val healthcheckPoller: HealthcheckPoller
) {
    suspend fun startRecording()
    suspend fun stopRecording()
    suspend fun pauseRecording()
    
    // Internal state machine
    private suspend fun recordingLoop()
    private suspend fun handleChunkReady(chunk: Chunk)
    private suspend fun uploadAndValidateChunk(chunk: Chunk): Boolean
}

enum class RecordingState {
    IDLE,
    RECORDING,
    PAUSED,
    UPLOADING,
    WAITING_HEALTHCHECK,
    ERROR
}
```

### 4.2 State Machine Logic

```
IDLE
  ↓ (startRecording)
RECORDING
  ├─ CameraManager captures frames → ChunkWriter buffers
  ├─ Chunk ready → UPLOADING
  └─ (stopRecording) → IDLE
  
UPLOADING
  ├─ ChunkUploader sends chunk
  ├─ Upload succeeds (202) → WAITING_HEALTHCHECK
  ├─ Upload fails (retryable) → retry, then UPLOADING or ERROR
  └─ Upload fails (non-retryable) → ERROR

WAITING_HEALTHCHECK
  ├─ Poll /chunks/{id}/status every 1s
  ├─ Status=healthy → Resume RECORDING with next chunk
  ├─ Status=unhealthy → Requeue current chunk for UPLOADING
  ├─ Timeout (60s) → ERROR
  └─ Network error → retry with exponential backoff

ERROR
  ├─ User can manually retry
  └─ (stopRecording) → IDLE
```

### 4.3 Error Handling & Retry

**Retryable Errors**:
- Network timeouts
- 5xx server errors
- Transient connection loss

**Non-Retryable Errors**:
- 4xx client errors (except timeout-related)
- Malformed chunk
- Authentication failure

**Retry Strategy**:
- Exponential backoff: `delay = base * (2 ^ attempt_count)`, max 60s
- Max 5 retries per chunk
- After 5 retries → buffer chunk in local queue, mark as ERROR

**Healthcheck Timeout**:
- Poll for 60 seconds
- If no response after 60s, assume processing failure
- Requeue chunk for upload (may indicate pipeline crash)

---

## 5. Pipeline Updates (Main Machine)

### 5.1 New HTTP Endpoints

#### POST /api/v1/chunks/upload
Receives multipart chunk upload. Validates basic structure (size, header JSON), stores to disk, queues for healthcheck.

```python
@app.post("/api/v1/chunks/upload")
async def upload_chunk(request: Request) -> dict:
    """
    - Parse multipart form
    - Validate chunk header JSON
    - Write to temp storage
    - Queue for healthcheck
    - Return 202 Accepted
    """
    try:
        chunk_id = validate_and_store_chunk(request)
        healthcheck_queue.put(chunk_id)
        return {"status": "received", "chunk_id": chunk_id}
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400
```

#### GET /api/v1/chunks/{chunk_id}/status
Returns healthcheck result (or pending if still checking).

```python
@app.get("/api/v1/chunks/{chunk_id}/status")
async def get_chunk_status(chunk_id: str) -> dict:
    """
    - Lookup chunk in healthcheck DB
    - Return validation result
    - If still pending, return 202 Accepted (client polls)
    """
    result = healthcheck_db.get(chunk_id)
    if result is None:
        return {"status": "pending", "message": "Still validating"}, 202
    return result.to_dict()
```

### 5.2 Healthcheck Worker

```python
class ChunkHealthchecker:
    """Validates chunks before processing."""
    
    def healthcheck(self, chunk_path: str) -> HealthcheckResult:
        """
        - Read chunk header & frame index
        - Verify frame count matches expected
        - Check CRC/SHA256
        - Detect timestamp or sequence gaps
        - Verify frames are in order
        - Log any anomalies
        - Return result
        """
        metadata = read_chunk_metadata(chunk_path)
        frames = read_frame_index(chunk_path)
        
        issues = []
        
        # Frame count
        if len(frames) != metadata['expected_frame_count']:
            issues.append(f"Expected {metadata['expected_frame_count']} frames, got {len(frames)}")
        
        # CRC
        if not verify_checksum(chunk_path, metadata['checksum']):
            issues.append("Checksum mismatch")
        
        # Sequence & timestamp gaps
        for i in range(1, len(frames)):
            seq_gap = frames[i]['sequence_number'] - frames[i-1]['sequence_number']
            if seq_gap != 1:
                issues.append(f"Sequence gap at frame {i}: {seq_gap - 1} frames missing")
            
            time_gap = frames[i]['timestamp_ms'] - frames[i-1]['timestamp_ms']
            expected_gap = 1000 / metadata['fps']
            if abs(time_gap - expected_gap) > expected_gap * 0.2:  # 20% tolerance
                issues.append(f"Timestamp anomaly at frame {i}")
        
        if issues:
            return HealthcheckResult.FAILED(issues)
        return HealthcheckResult.OK()
```

### 5.3 Processing Queue

Update the main detection pipeline to consume from a **validated chunk queue**:

```python
class ChunkProcessor:
    """Main processing loop — only runs on healthy chunks."""
    
    def process_chunks(self):
        while True:
            chunk_id = validated_chunk_queue.get()  # Blocks until available
            chunk_path = resolve_chunk_path(chunk_id)
            
            try:
                # Run existing YOLO pipeline
                detections = self.run_detection(chunk_path)
                events = self.run_event_logic(detections)
                self.aggregate_stats(events)
            except Exception as e:
                logger.error(f"Processing failed for {chunk_id}: {e}")
```

---

## 6. Network & Performance Considerations

### 6.1 Chunk Size Optimization

**Default**: 30 frames @ 30 FPS = 1 second

**Rationale**:
- **Small chunks** (10 frames): Lower latency, more frequent validation, higher overhead
- **Medium chunks** (30 frames): 1s duration, good balance
- **Large chunks** (90 frames): Fewer uploads, but longer wait on healthcheck failure

**Recommendation**: Start with 30 frames, make tunable via `config.yaml`.

```yaml
# config.yaml
android_frontend:
  chunk_size_frames: 30
  chunk_timeout_ms: 60000
  upload_timeout_ms: 30000
  healthcheck_poll_interval_ms: 1000
  max_upload_retries: 5
  network_buffer_chunks: 2  # Keep 2 chunks in local buffer
```

### 6.2 Bandwidth

Assuming:
- Resolution: 1920×1080
- Encoding: H.264 @ 5-8 Mbps
- Chunk duration: 1 second
- Chunk size: ~625–1000 KB

**Upload time** (10 Mbps WiFi): ~0.5–1 second
**Healthcheck latency**: 1–2 seconds (local inference)
**Total cycle**: ~2–3 seconds per chunk

This is acceptable for a live game (realtime not critical).

### 6.3 Graceful Backpressure

If uploads fall behind:
- App buffers up to 2 chunks locally
- If 3rd chunk completes and 1st still uploading → **block camera thread** (drop current frame gracefully) or **slow down recording**?
- **Recommendation**: Slow down recording (reduce FPS) rather than drop frames silently

---

## 7. Data Storage & Cleanup

### 7.1 Local App Storage

```
/data/data/com.basketballcv.recorder/
├── chunks/
│   ├── camera_0_chunk_0001.tmp  (in-progress)
│   ├── camera_0_chunk_0002.confirmed  (uploaded, waiting healthcheck)
│   └── camera_0_chunk_0000.uploaded  (processed)
├── metadata.json  (app state: last chunk ID, last healthcheck status, etc.)
└── logs/
    └── recording_2026_04_21.log
```

### 7.2 Pipeline Storage

```
/data/basketball-cv/chunks/
├── received/
│   └── camera_0_chunk_*.received
├── validated/
│   └── camera_0_chunk_*.validated
├── processed/
│   └── camera_0_chunk_*.processed
└── failed/
    └── camera_0_chunk_*.failed (reason.txt)
```

**Cleanup**:
- Delete `received/` chunks older than 24 hours (after validation)
- Archive `processed/` chunks after 7 days
- Retain `failed/` for debugging (notify app via API that chunk is unrecoverable)

---

## 8. UI/UX Design

### 8.1 Main Recording Screen

```
┌─────────────────────────────────────┐
│  Basketball CV Recorder             │
├─────────────────────────────────────┤
│                                     │
│   [Camera Preview]                  │
│                                     │
├─────────────────────────────────────┤
│ Recording: 00:02:34                 │
│ Frames captured: 4680               │
│ Frames dropped: 0                   │
│ Chunks uploaded: 156                │
│ Chunks pending: 1 (healthcheck)     │
├─────────────────────────────────────┤
│ Connectivity: ✓ WiFi (10 Mbps)      │
│ Battery: 45%                        │
│ Temperature: Normal                 │
├─────────────────────────────────────┤
│  [RECORD] [PAUSE] [STOP]            │
└─────────────────────────────────────┘
```

### 8.2 Status Indicators

- **Green**: Recording + all chunks uploaded & healthy
- **Yellow**: Chunk pending healthcheck or retrying upload
- **Red**: Network error or unhealthy chunk (user notified, manual retry option)

### 8.3 Settings

```
Camera
├─ FPS: [30] fps
├─ Resolution: [1920x1080]
└─ Encoding: [H.264 / VP9 / Raw]

Network
├─ Backend URL: [input field]
├─ Chunk size: [30] frames
├─ Upload timeout: [30] seconds
├─ Max retries: [5]
└─ Disable upload on cellular: [Toggle]

Debug
├─ Enable logging: [Toggle]
├─ Log level: [Debug / Info / Warn]
└─ Export logs: [Button]
```

---

## 9. Testing & Validation

### 9.1 Unit Tests (Android)

- `CameraManagerTest`: Mock frame capture, verify sequence numbers
- `ChunkWriterTest`: Verify frame buffering, chunk finalization
- `UploaderTest`: Mock HTTP responses, test retry logic
- `HealthcheckPollerTest`: Mock API responses, test timeout handling
- `OrchestratorTest`: State machine transitions, error recovery

### 9.2 Integration Tests (Android + Pipeline)

- **Happy path**: Record 5 seconds (150 frames), verify all chunks processed
- **Network dropout**: Simulate WiFi disconnect during upload, verify resumption
- **Unhealthy chunk**: Inject dropped frames, verify pipeline rejects and app retries
- **Pipeline slowdown**: Delay healthcheck response, verify app doesn't timeout or retry prematurely

### 9.3 Load Tests (Pipeline)

- **Concurrent uploads**: 3 cameras uploading simultaneously
- **Healthcheck CPU**: Verify healthcheck doesn't block processing
- **Disk I/O**: Measure chunk read/write throughput

### 9.4 Network Tests

- **Latency**: Simulate 50–200 ms RTT
- **Bandwidth**: Simulate 5–50 Mbps
- **Jitter**: Simulate packet loss (1–5%)
- **Timeout**: Simulate server hang (>60s)

---

## 10. Scalability & Future Work

### 10.1 Multi-Camera Support

Current spec assumes single camera per app instance. For multiple cameras:

**Option A**: One app instance per camera (separate devices or processes)
- Simpler implementation
- No coordination overhead
- Scales naturally

**Option B**: One app, multiple camera streams
- Requires thread-safe camera/chunk management
- Shared network connection
- More complex state machine

**Recommendation**: Start with Option A (one app per camera). Expand to Option B if needed.

### 10.2 Dynamic FPS/Resolution

Allow pipeline to request the app adjust FPS or resolution based on network conditions:

```
POST /api/v1/camera/adjust
{
  "target_fps": 15,
  "target_resolution": "1280x720"
}
```

### 10.3 Adaptive Chunk Size

If healthcheck frequently detects dropped frames, automatically reduce chunk size (more frequent uploads = higher chance of detecting issues early).

### 10.4 Local Frame Logging

Option to write raw frame data locally (for offline debugging) with configurable retention.

---

## 11. Configuration

### 11.1 config.yaml (Main Machine)

```yaml
android_frontend:
  enabled: true
  listen_address: "0.0.0.0"
  listen_port: 8000
  chunk_storage_path: "/data/basketball-cv/chunks"
  healthcheck_timeout_ms: 60000
  healthcheck_poll_interval_ms: 1000
  max_concurrent_healthchecks: 4
  
  cleanup:
    delete_received_after_days: 1
    delete_processed_after_days: 7
    archive_processed: true

logging:
  level: "INFO"
  file: "/data/basketball-cv/logs/frontend.log"
```

### 11.2 App Preferences (Android)

Stored in Android SharedPreferences or DataStore:

```
backend_url: "http://192.168.1.100:8000"
camera_fps: 30
camera_resolution: "1920x1080"
chunk_size_frames: 30
upload_timeout_ms: 30000
max_retries: 5
disable_cellular: false
logging_enabled: false
```

---

## 12. API Reference

### Chunk Upload
```
POST /api/v1/chunks/upload
Content-Type: multipart/form-data

metadata: {...}  // JSON chunk header
video: <binary>  // Frame data
checksum: "..."  // CRC or SHA256

Response:
202 Accepted
{
  "status": "received",
  "chunk_id": "camera_0_chunk_1234",
  "timestamp_received_ms": 1713700124500
}
```

### Chunk Status
```
GET /api/v1/chunks/{chunk_id}/status

Response (Healthy):
200 OK
{
  "chunk_id": "camera_0_chunk_1234",
  "status": "healthy",
  "validation": {...},
  "message": "Chunk passed healthcheck."
}

Response (Unhealthy):
400 Bad Request
{
  "chunk_id": "camera_0_chunk_1234",
  "status": "unhealthy",
  "validation": {...},
  "missing_sequences": [10, 23],
  "message": "Chunk has 2 dropped frames."
}

Response (Pending):
202 Accepted
{
  "status": "pending",
  "message": "Still validating..."
}
```

### Camera Control (Future)
```
POST /api/v1/camera/adjust
{
  "target_fps": 15,
  "target_resolution": "1280x720"
}

Response:
200 OK
{
  "status": "acknowledged",
  "message": "App has been notified. Changes may take effect next chunk."
}
```

---

## 13. Dependencies

### Android App

**Core**:
- Android SDK 24+ (API 24)
- AndroidX libraries (lifecycle, work, datastore)
- Kotlin Coroutines

**Media**:
- `androidx.camera:camera-camera2` (CameraX)
- `com.google.android.exoplayer:exoplayer` (optional, if needed for testing)

**Networking**:
- OkHttp 4+ (with interceptors for retry logic)
- Retrofit (type-safe HTTP client)
- Moshi (JSON serialization)

**Storage**:
- AndroidX Datastore (preferences)
- File I/O (standard Java)

**Logging**:
- Timber (logging facade)

**Testing**:
- JUnit 4
- Mockito
- Espresso (UI testing)
- Robolectric (unit testing on JVM)

### Pipeline (Main Machine)

**Framework**:
- FastAPI (async HTTP server)

**Validation**:
- pydantic (data validation)

**Storage**:
- stdlib pathlib

**Hashing**:
- hashlib (CRC32, SHA256)

**Logging**:
- stdlib logging

---

## 14. Security Considerations

### 14.1 Authentication (MVP: None, Future: API Key)

For MVP, assume app and pipeline are on same local network. Future: implement simple API key auth.

```
Header: X-API-Key: <secret>
```

### 14.2 Data Integrity

- Every chunk includes CRC32 or SHA256
- Pipeline verifies checksum before processing
- Frame sequence numbers prevent out-of-order reassembly

### 14.3 Network Security

- Assume WiFi is trusted (indoor gym)
- Future: HTTPS if deployed to untrusted networks

---

## 15. Summary & Next Steps

### Implementation Priority

1. **Phase 1A (Week 1)**:
   - Android app: Camera capture + frame buffering
   - Pipeline: HTTP upload endpoint
   - Basic chunk health checks

2. **Phase 1B (Week 2)**:
   - Android app: Upload orchestration + retry logic
   - Pipeline: Healthcheck worker + validation
   - End-to-end test with single chunk

3. **Phase 1C (Week 3)**:
   - UI/UX polish
   - Performance testing & optimization
   - Error handling + graceful degradation

4. **Phase 1D (Week 4)**:
   - Multi-camera testing
   - Network stress tests
   - Documentation & handoff

### Key Design Decisions

✅ **Sequential validation**: Ensures no silent frame loss; trade-off is latency (~2–3s per chunk)

✅ **Bounded backpressure**: App waits for ACK before uploading next chunk; simple to implement, prevents buffer explosion

✅ **Correctness over speed**: Focus on frame integrity; performance can be scaled vertically later

✅ **HTTP + polling**: Simpler than gRPC/WebSocket for MVP; adequate latency for live game scenario

---

## Appendix: Best Practices for Video Streaming Pipelines

### A.1 Reference Architectures

1. **AWS Kinesis Video Streams**: Uses fragment-based payloads (similar to our chunks) with automatic failover
2. **RTMP Ingest (Twitch/YouTube)**: Real-time pipeline with adaptive bitrate; not suited for our correctness-first approach
3. **DASH/HLS Chunking**: Media streaming uses fixed-duration segments with manifests; relevant for Phase 1.5 (scoreboard overlay)

### A.2 Lessons Learned

- **Always include sequence numbers**: Essential for detecting dropped frames
- **Checksums are cheap**: Include CRC32 or SHA256 on every chunk; CPU cost is negligible
- **Timeouts are critical**: Set reasonable poll/upload timeouts; avoid infinite waits
- **Graceful degradation > fast failure**: Better to slow down than lose data silently
- **Log everything**: Frame counts, timestamps, gaps, retries — invaluable for debugging

### A.3 Common Pitfalls

❌ Assuming network reliability without validation
❌ Silent drops (no error message to user)
❌ Unbounded retries (can exhaust resources)
❌ No backpressure (buffer overflow)
❌ Tight coupling between app and pipeline (hard to debug)

