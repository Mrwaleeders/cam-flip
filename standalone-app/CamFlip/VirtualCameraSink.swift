import CoreMedia
import CoreMediaIO
import Foundation

/// Writes processed frames into the Cam Flip Camera's sink stream.
/// Sample buffers are handed over by reference (IOSurface), so nothing is copied or compressed.
final class VirtualCameraSink {
    private var deviceID: CMIOObjectID = 0
    private var streamID: CMIOStreamID = 0
    private var queue: CMSimpleQueue?
    private let lock = NSLock()

    var isConnected: Bool {
        lock.lock(); defer { lock.unlock() }
        return queue != nil
    }

    /// Returns true once connected. Safe to call repeatedly.
    @discardableResult
    func connect() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if queue != nil { return true }

        guard let device = Self.findDevice() else { return false }
        let streams = Self.streams(of: device)
        guard streams.count >= 2 else { return false }
        let sink = streams[1]

        var unmanagedQueue: Unmanaged<CMSimpleQueue>?
        let status = CMIOStreamCopyBufferQueue(sink, { _, _, _ in }, nil, &unmanagedQueue)
        guard status == noErr, let unmanagedQueue else { return false }

        let startStatus = CMIODeviceStartStream(device, sink)
        guard startStatus == noErr else {
            unmanagedQueue.release()
            return false
        }
        deviceID = device
        streamID = sink
        queue = unmanagedQueue.takeRetainedValue()
        return true
    }

    func disconnect() {
        lock.lock(); defer { lock.unlock() }
        guard queue != nil else { return }
        CMIODeviceStopStream(deviceID, streamID)
        queue = nil
    }

    func send(_ sampleBuffer: CMSampleBuffer) {
        lock.lock(); defer { lock.unlock() }
        guard let queue else { return }
        // Drop rather than queue up: latency matters more than a stray frame.
        guard CMSimpleQueueGetCount(queue) < CMSimpleQueueGetCapacity(queue) else { return }
        CMSimpleQueueEnqueue(queue, element: Unmanaged.passRetained(sampleBuffer).toOpaque())
    }

    // MARK: CoreMediaIO lookup

    static func findDevice() -> CMIOObjectID? {
        let uid = CamFlipShared.deviceUUID.uuidString
        for device in objects(of: CMIOObjectID(kCMIOObjectSystemObject), selector: kCMIOHardwarePropertyDevices) {
            if deviceUID(device) == uid { return device }
        }
        return nil
    }

    static func streams(of device: CMIOObjectID) -> [CMIOStreamID] {
        objects(of: device, selector: kCMIODevicePropertyStreams)
    }

    private static func objects(of object: CMIOObjectID, selector: Int) -> [CMIOObjectID] {
        var address = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(selector),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
        )
        var dataSize: UInt32 = 0
        guard CMIOObjectGetPropertyDataSize(object, &address, 0, nil, &dataSize) == noErr, dataSize > 0 else { return [] }
        let count = Int(dataSize) / MemoryLayout<CMIOObjectID>.size
        var ids = [CMIOObjectID](repeating: 0, count: count)
        var used: UInt32 = 0
        guard CMIOObjectGetPropertyData(object, &address, 0, nil, dataSize, &used, &ids) == noErr else { return [] }
        return Array(ids.prefix(Int(used) / MemoryLayout<CMIOObjectID>.size))
    }

    private static func deviceUID(_ device: CMIOObjectID) -> String? {
        var address = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(kCMIODevicePropertyDeviceUID),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
        )
        var uid: Unmanaged<CFString>?
        var used: UInt32 = 0
        let size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard CMIOObjectGetPropertyData(device, &address, 0, nil, size, &used, &uid) == noErr,
              let uid else { return nil }
        return uid.takeRetainedValue() as String
    }
}
