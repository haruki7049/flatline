//! Loader for SafeTensors weight files: header parsing and tensor offset resolution.
//! Layout: [u64 LE header length][JSON header][payload]. Tensors are zero-copy views
//! into the payload.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

comptime {
    // Tensor data is reinterpreted in place as native f32.
    if (builtin.cpu.arch.endian() != .little) @compileError("SafeTensors loading requires a little-endian target");
}

pub const max_rank = 4;

/// A borrowed F32 tensor. Valid as long as the backing bytes are alive.
pub const Tensor = struct {
    data: []const f32,
    shape_buf: [max_rank]usize,
    rank: usize,

    pub fn shape(self: *const Tensor) []const usize {
        return self.shape_buf[0..self.rank];
    }
};

/// Parsed view of SafeTensors bytes. Does not own the bytes.
pub const SafeTensors = struct {
    header_json: []const u8,
    payload: []const u8,
    header: std.json.Parsed(std.json.Value),

    pub fn parse(allocator: Allocator, bytes: []const u8) !SafeTensors {
        if (bytes.len < 8) return error.InvalidFile;

        // 8-byte little-endian header length
        const header_len = std.math.cast(usize, std.mem.readInt(u64, bytes[0..8], .little)) orelse
            return error.InvalidFile;
        if (header_len > bytes.len - 8) return error.InvalidFile;
        const header_json = bytes[8 .. 8 + header_len];

        const header = try std.json.parseFromSlice(std.json.Value, allocator, header_json, .{});
        errdefer header.deinit();
        if (header.value != .object) return error.InvalidHeader;

        return .{
            .header_json = header_json,
            .payload = bytes[8 + header_len ..],
            .header = header,
        };
    }

    pub fn deinit(self: *SafeTensors) void {
        self.header.deinit();
    }

    /// Resolves `name` to its F32 data in the payload.
    pub fn tensor(self: *const SafeTensors, name: []const u8) !Tensor {
        const entry = self.header.value.object.get(name) orelse return error.TensorNotFound;
        if (entry != .object) return error.InvalidHeader;

        const dtype = entry.object.get("dtype") orelse return error.InvalidHeader;
        if (dtype != .string) return error.InvalidHeader;
        if (!std.mem.eql(u8, dtype.string, "F32")) return error.UnsupportedDtype;

        var result: Tensor = .{ .data = &.{}, .shape_buf = undefined, .rank = 0 };
        const shape = try getArray(entry, "shape");
        if (shape.len > max_rank) return error.UnsupportedRank;
        var num_elements: usize = 1;
        for (shape, 0..) |dim, i| {
            result.shape_buf[i] = try toUsize(dim);
            num_elements = std.math.mul(usize, num_elements, result.shape_buf[i]) catch return error.InvalidHeader;
        }
        result.rank = shape.len;

        const offsets = try getArray(entry, "data_offsets");
        if (offsets.len != 2) return error.InvalidHeader;
        const start = try toUsize(offsets[0]);
        const end = try toUsize(offsets[1]);
        if (start > end or end > self.payload.len) return error.InvalidHeader;

        const bytes = self.payload[start..end];
        if (bytes.len != num_elements * @sizeOf(f32)) return error.InvalidHeader;
        if (!std.mem.isAligned(@intFromPtr(bytes.ptr), @alignOf(f32))) return error.MisalignedTensor;

        const ptr: [*]const f32 = @ptrCast(@alignCast(bytes.ptr));
        result.data = ptr[0..num_elements];
        return result;
    }

    /// Like `tensor`, with the name built from a format string.
    pub fn tensorFmt(self: *const SafeTensors, comptime fmt: []const u8, args: anytype) !Tensor {
        var buf: [256]u8 = undefined;
        return self.tensor(try std.fmt.bufPrint(&buf, fmt, args));
    }

    fn getArray(entry: std.json.Value, key: []const u8) ![]const std.json.Value {
        const value = entry.object.get(key) orelse return error.InvalidHeader;
        if (value != .array) return error.InvalidHeader;
        return value.array.items;
    }

    fn toUsize(value: std.json.Value) !usize {
        if (value != .integer) return error.InvalidHeader;
        return std.math.cast(usize, value.integer) orelse error.InvalidHeader;
    }
};

/// A SafeTensors file mapped into memory (zero-copy).
pub const Model = struct {
    mapped: []align(std.heap.page_size_min) const u8,
    weights: SafeTensors,

    pub fn open(io: std.Io, allocator: Allocator, path: []const u8) !Model {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        // The mapping stays valid after the file is closed.
        defer file.close(io);

        const file_size = std.math.cast(usize, (try file.stat(io)).size) orelse return error.InvalidFile;
        if (file_size < 8) return error.InvalidFile;

        const mapped = try std.posix.mmap(
            null,
            file_size,
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mapped);

        return .{
            .mapped = mapped,
            .weights = try SafeTensors.parse(allocator, mapped),
        };
    }

    pub fn close(self: *Model) void {
        self.weights.deinit();
        std.posix.munmap(self.mapped);
    }
};

/// Builds SafeTensors bytes for tests, padding the header so the payload stays 4-byte aligned.
fn buildTestFile(buf: []align(4) u8, header: []const u8, payload: []const f32) []align(4) u8 {
    const header_len = std.mem.alignForward(usize, 8 + header.len, 4) - 8;
    std.mem.writeInt(u64, buf[0..8], header_len, .little);
    @memcpy(buf[8..][0..header.len], header);
    @memset(buf[8 + header.len .. 8 + header_len], ' ');
    const payload_bytes = std.mem.sliceAsBytes(payload);
    @memcpy(buf[8 + header_len ..][0..payload_bytes.len], payload_bytes);
    return buf[0 .. 8 + header_len + payload_bytes.len];
}

test "SafeTensors resolves tensor offsets and shapes" {
    var buf: [256]u8 align(4) = undefined;
    const bytes = buildTestFile(&buf,
        \\{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1,2],"data_offsets":[8,16]}}
    , &.{ 1.0, 2.0, 3.0, 4.0 });

    var st = try SafeTensors.parse(std.testing.allocator, bytes);
    defer st.deinit();

    const a = try st.tensor("a");
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, a.data);
    try std.testing.expectEqualSlices(usize, &.{2}, a.shape());

    const b = try st.tensorFmt("{s}", .{"b"});
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 4.0 }, b.data);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, b.shape());

    try std.testing.expectError(error.TensorNotFound, st.tensor("missing"));
}

test "SafeTensors rejects inconsistent entries" {
    var buf: [256]u8 align(4) = undefined;
    const bytes = buildTestFile(&buf,
        \\{"short":{"dtype":"F32","shape":[3],"data_offsets":[0,8]},"oob":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"half":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}
    , &.{ 1.0, 2.0 });

    var st = try SafeTensors.parse(std.testing.allocator, bytes);
    defer st.deinit();

    try std.testing.expectError(error.InvalidHeader, st.tensor("short"));
    try std.testing.expectError(error.InvalidHeader, st.tensor("oob"));
    try std.testing.expectError(error.UnsupportedDtype, st.tensor("half"));
}

test "SafeTensors rejects a truncated header" {
    var buf: [16]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], 100, .little);
    try std.testing.expectError(error.InvalidFile, SafeTensors.parse(std.testing.allocator, &buf));
}
