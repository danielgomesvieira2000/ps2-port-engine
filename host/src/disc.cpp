#include "ps2host/disc.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace ps2host {

namespace {

u32 Le32(const u8* p) { return u32(p[0]) | u32(p[1]) << 8 | u32(p[2]) << 16 | u32(p[3]) << 24; }

std::string Upper(std::string s) {
    for (char& ch : s) ch = char(std::toupper(static_cast<unsigned char>(ch)));
    return s;
}

// A directory record's name without its version (";1") and trailing dot.
std::string Bare(std::string name) {
    const auto semicolon = name.find(';');
    if (semicolon != std::string::npos) name.resize(semicolon);
    if (!name.empty() && name.back() == '.') name.pop_back();
    return Upper(name);
}

}  // namespace

Disc::~Disc() {
    if (file_) std::fclose(file_);
}

namespace {

// The image a .cue sheet names in its first FILE line, beside the sheet.
std::string CueImage(const std::string& cue_path) {
    std::FILE* cue = std::fopen(cue_path.c_str(), "rb");
    if (!cue) return "";
    std::string text;
    char chunk[4096];
    for (std::size_t n; (n = std::fread(chunk, 1, sizeof chunk, cue)) > 0;) text.append(chunk, n);
    std::fclose(cue);
    for (std::size_t at = 0; at < text.size();) {
        std::size_t end = text.find_first_of("\r\n", at);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(at, end - at);
        at = end + 1;
        line.erase(0, line.find_first_not_of(" \t"));
        if (Upper(line.substr(0, 5)) != "FILE ") continue;
        std::string name = line.substr(5);
        name.erase(0, name.find_first_not_of(" \t"));
        if (!name.empty() && name[0] == '"') {
            const std::size_t close = name.find('"', 1);
            name = name.substr(1, close == std::string::npos ? std::string::npos : close - 1);
        } else {
            name = name.substr(0, name.find_first_of(" \t"));
        }
        const std::size_t slash = cue_path.find_last_of("/\\");
        return slash == std::string::npos ? name : cue_path.substr(0, slash + 1) + name;
    }
    return "";
}

const u8 kSync[12] = {0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00};

u8 ToBcd(u32 v) { return u8((v / 10) << 4 | (v % 10)); }

void (*g_on_open)() = nullptr;

}  // namespace

void Disc::OnOpen(void (*hook)()) { g_on_open = hook; }

bool Disc::Open(const std::string& path) {
    if (file_) std::fclose(file_);
    file_ = nullptr;
    memory_.clear();
    std::string image = path;
    if (image.size() > 4 && Upper(image.substr(image.size() - 4)) == ".CUE") {
        image = CueImage(path);
        if (image.empty()) return false;
    }
    file_ = std::fopen(image.c_str(), "rb");
    if (!file_) return false;
    _fseeki64(file_, 0, SEEK_END);
    size_ = u64(_ftelli64(file_));
    DetectFormat();
    opening_ = true;                    // (AddFiles then only takes the folder)
    if (g_on_open) g_on_open();
    opening_ = false;
    ApplyFiles();
    return true;
}

void Disc::Use(std::vector<u8> image) {
    memory_ = std::move(image);
    size_ = memory_.size();
    DetectFormat();
}

// A raw image: every sector starts with the sync pattern, and sector 16 - the
// volume descriptor, on every PS2 disc - says the mode.
void Disc::DetectFormat() {
    data_offset_ = 0;
    u8 head[16];
    if (size_ >= 17 * u64(kRawSectorSize) && ReadBytes(16 * u64(kRawSectorSize), 16, head) &&
        std::memcmp(head, kSync, 12) == 0 && (head[15] == 1 || head[15] == 2))
        data_offset_ = head[15] == 1 ? 16 : 24;
    sectors_ = u32(size_ / (data_offset_ ? kRawSectorSize : kSectorSize));
}

bool Disc::IsCd() const {
    // A CD holds at most about 360,000 sectors (80 minutes); a DVD image is raw-less.
    return data_offset_ != 0 || sectors_ <= 360000;
}

bool Disc::ReadBytes(u64 offset, u64 length, u8* out) {
    if (offset + length > size_) return false;
    if (!memory_.empty()) {
        std::memcpy(out, memory_.data() + offset, size_t(length));
        return true;
    }
    if (!file_ || _fseeki64(file_, s64(offset), SEEK_SET) != 0) return false;
    return std::fread(out, 1, size_t(length), file_) == length;
}

bool Disc::Read(u32 lsn, u32 count, u8* out) {
    if (patched_.empty() && extras_.empty()) return BaseRead(lsn, count, out);
    // The overlay's: sector by sector, a run of the image's own read at once.
    for (u32 i = 0; i < count;) {
        u8* dst = out + std::size_t(i) * kSectorSize;
        if (SectorAt(lsn + i, dst)) {
            ++i;
            continue;
        }
        u32 run = 1;
        while (i + run < count) {
            const u32 next = lsn + i + run;
            bool overlaid = next >= sectors_;
            for (const auto& p : patched_) overlaid = overlaid || p.first == next;
            if (overlaid) break;
            ++run;
        }
        if (!BaseRead(lsn + i, run, dst)) return false;
        i += run;
    }
    return true;
}

bool Disc::BaseRead(u32 lsn, u32 count, u8* out) {
    if (u64(lsn) + count > sectors_) return false;
    if (!data_offset_) return ReadBytes(u64(lsn) * kSectorSize, u64(count) * kSectorSize, out);
    // Raw: read the whole run at once, then keep each sector's data.
    std::vector<u8> raw(std::size_t(count) * kRawSectorSize);
    if (!ReadBytes(u64(lsn) * kRawSectorSize, raw.size(), raw.data())) return false;
    for (u32 i = 0; i < count; ++i)
        std::memcpy(out + std::size_t(i) * kSectorSize, &raw[std::size_t(i) * kRawSectorSize + data_offset_],
                    kSectorSize);
    return true;
}

bool Disc::ReadPattern(u32 lsn, u32 count, u32 pattern, u8* out) {
    if (pattern != 1 && pattern != 2) return Read(lsn, count, out);
    if (u64(lsn) + count > sectors_) return false;
    const u32 size = PatternSize(pattern);
    // 2340 starts at the header (offset 12), 2328 after the subheader (24).
    const u32 from = pattern == 2 ? 12 : 24;
    std::vector<u8> raw(kRawSectorSize);
    for (u32 i = 0; i < count; ++i) {
        u8* dst = out + std::size_t(i) * size;
        if (data_offset_) {
            if (!ReadBytes(u64(lsn + i) * kRawSectorSize, kRawSectorSize, raw.data())) return false;
            std::memcpy(dst, &raw[from], size);
            continue;
        }
        // An .iso: a Mode 2 Form 1 sector made around its data.
        std::memset(raw.data(), 0, raw.size());
        const u32 address = lsn + i + 150;          // the 2-second lead-in
        raw[12] = ToBcd(address / 75 / 60);
        raw[13] = ToBcd(address / 75 % 60);
        raw[14] = ToBcd(address % 75);
        raw[15] = 2;
        raw[18] = raw[22] = 0x08;                   // subheader submode: data
        if (!Read(lsn + i, 1, &raw[24])) return false;
        std::memcpy(dst, &raw[from], size);
    }
    return true;
}

bool Disc::Find(const std::string& path, File* out) {
    // The primary volume descriptor, sector 16, holds the root directory's record.
    std::vector<u8> sector(kSectorSize);
    if (!Read(16, 1, sector.data()) || sector[0] != 1 || std::memcmp(&sector[1], "CD001", 5) != 0)
        return false;
    File directory{Le32(&sector[156 + 2]), Le32(&sector[156 + 10])};
    std::string rest = path;
    for (char& ch : rest) if (ch == '/') ch = '\\';
    while (!rest.empty()) {
        while (!rest.empty() && rest[0] == '\\') rest.erase(0, 1);
        if (rest.empty()) break;
        const auto end = rest.find('\\');
        const std::string part = Bare(rest.substr(0, end));
        rest = end == std::string::npos ? "" : rest.substr(end);
        // Scan the directory's records for this part.
        const u32 sectors = (directory.size + kSectorSize - 1) / kSectorSize;
        std::vector<u8> data(size_t(sectors) * kSectorSize);
        if (!Read(directory.lsn, sectors, data.data())) return false;
        bool found = false;
        for (u32 at = 0; at < directory.size;) {
            const u8 length = data[at];
            if (length == 0) {                  // records never cross a sector boundary
                at = (at / kSectorSize + 1) * kSectorSize;
                continue;
            }
            const u8 name_length = data[at + 32];
            const std::string name(reinterpret_cast<const char*>(&data[at + 33]), name_length);
            if (name_length > 1 || (name_length == 1 && name[0] > 1)) {
                if (Bare(name) == part) {
                    directory = File{Le32(&data[at + 2]), Le32(&data[at + 10])};
                    found = true;
                    break;
                }
            }
            at += length;
        }
        if (!found) return false;
    }
    *out = directory;
    return true;
}

bool Disc::ReadFile(const std::string& path, std::vector<u8>* out) {
    File file;
    if (!Find(path, &file)) return false;
    out->assign(size_t(file.size + kSectorSize - 1) / kSectorSize * kSectorSize, 0);
    if (!Read(file.lsn, u32(out->size() / kSectorSize), out->data())) return false;
    out->resize(file.size);
    return true;
}

std::string Disc::BootPath() {
    std::vector<u8> cnf;
    if (!ReadFile("SYSTEM.CNF", &cnf)) return "";
    const std::string text(cnf.begin(), cnf.end());
    for (std::size_t at = 0; at < text.size();) {
        std::size_t end = text.find_first_of("\r\n", at);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(at, end - at);
        at = end + 1;
        // BOOT2 = cdrom0:\SLUS_210.50;1
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) continue;
        std::string key = line.substr(0, equals);
        key.erase(key.find_last_not_of(" \t") + 1);
        if (Upper(key) != "BOOT2") continue;
        std::string value = line.substr(equals + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        const std::size_t colon = value.find(':');
        return colon == std::string::npos ? value : value.substr(colon + 1);
    }
    return "";
}

// ---------------------------------------------------------------- the overlay (AddFiles)

namespace {

void PutBoth32(u8* p, u32 v) {              // ISO 9660's both-endian word
    p[0] = u8(v); p[1] = u8(v >> 8); p[2] = u8(v >> 16); p[3] = u8(v >> 24);
    p[4] = u8(v >> 24); p[5] = u8(v >> 16); p[6] = u8(v >> 8); p[7] = u8(v);
}

// A directory record: a file ("NAME.EXT;1") or a directory (flags 2).
std::vector<u8> Record(const std::string& name, u32 lsn, u32 size, bool directory) {
    const std::size_t length = (33 + name.size() + 1) & ~std::size_t(1);
    std::vector<u8> r(length, 0);
    r[0] = u8(length);
    PutBoth32(&r[2], lsn);
    PutBoth32(&r[10], size);
    const u8 date[7] = {104, 9, 7, 12, 0, 0, 0};        // 2004-09-07 12:00, the game's release
    std::memcpy(&r[18], date, 7);
    r[25] = directory ? 2 : 0;
    r[28] = 1;                                          // volume sequence number 1, both-endian
    r[31] = 1;
    r[32] = u8(name.size());
    std::memcpy(&r[33], name.data(), name.size());
    return r;
}

}  // namespace

void Disc::AddFiles(const std::string& folder) {
    if (std::find(folders_.begin(), folders_.end(), folder) == folders_.end()) folders_.push_back(folder);
    if (IsOpen() && !opening_) ApplyFiles();
}

bool Disc::SectorAt(u32 lsn, u8* out) {
    for (const auto& p : patched_)
        if (p.first == lsn) {
            std::memcpy(out, p.second.data(), kSectorSize);
            return true;
        }
    for (const Extra& e : extras_)
        if (lsn >= e.lsn && lsn < e.lsn + e.sectors) {
            std::memset(out, 0, kSectorSize);
            if (std::FILE* f = std::fopen(e.path.c_str(), "rb")) {
                _fseeki64(f, s64(lsn - e.lsn) * kSectorSize, SEEK_SET);
                (void)std::fread(out, 1, kSectorSize, f);
                std::fclose(f);
            }
            return true;
        }
    return false;
}

u8* Disc::Patch(u32 lsn) {
    for (auto& p : patched_)
        if (p.first == lsn) return p.second.data();
    std::vector<u8> sector(kSectorSize, 0);
    if (!SectorAt(lsn, sector.data()) && lsn < sectors_) BaseRead(lsn, 1, sector.data());
    patched_.emplace_back(lsn, std::move(sector));
    return patched_.back().second.data();
}

void Disc::ApplyFiles() {
    if (folders_.empty() || !IsOpen()) return;
    patched_.clear();
    extras_.clear();
    next_virtual_ = (sectors_ + 0xFFF) & ~0xFFFu;      // past the image, a round number
    namespace fs = std::filesystem;
    for (const std::string& folder : folders_) {
        std::error_code error;
        if (!fs::is_directory(fs::u8path(folder), error)) {
            std::fprintf(stderr, "[disc] no folder %s to lay over the disc\n", folder.c_str());
            continue;
        }
        std::vector<std::pair<std::string, fs::path>> files;
        for (auto it = fs::recursive_directory_iterator(fs::u8path(folder), error);
             !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
            if (!it->is_regular_file(error)) continue;
            const std::string relative = fs::relative(it->path(), fs::u8path(folder), error).generic_u8string();
            files.emplace_back(relative, it->path());
        }
        std::sort(files.begin(), files.end());
        u32 added = 0;
        for (const auto& [relative, path] : files) {
            const auto size = fs::file_size(path, error);
            if (error || size > 0xFFFFFFFFull) continue;
            if (AddFile(relative, path.u8string(), u32(size))) ++added;
            else std::fprintf(stderr, "[disc] %s: not laid over the disc\n", relative.c_str());
        }
        std::fprintf(stderr, "[disc] %u files from %s laid over the disc\n", added, folder.c_str());
    }
}

/* One file of the overlay: its directories found (or made) from the root,
   then its record changed (a file of the disc's) or added. */
bool Disc::AddFile(const std::string& relative, const std::string& host_path, u32 size) {
    std::vector<std::string> parts;
    for (std::size_t at = 0; at <= relative.size();) {
        std::size_t end = relative.find('/', at);
        if (end == std::string::npos) end = relative.size();
        if (end > at) parts.push_back(Upper(relative.substr(at, end - at)));
        at = end + 1;
    }
    if (parts.empty()) return false;
    // A directory, and where its own record is: (sector, offset) - the
    // root's in the volume descriptor.
    struct Dir {
        u32 lsn = 0, size = 0;
        u32 record_lsn = 0, record_at = 0;
    };
    std::vector<u8> sector(kSectorSize);
    if (!Read(16, 1, sector.data()) || sector[0] != 1 || std::memcmp(&sector[1], "CD001", 5) != 0) return false;
    Dir dir{Le32(&sector[156 + 2]), Le32(&sector[156 + 10]), 16, 156};
    // A record of `d` by name: its sector and offset.
    auto find = [&](const Dir& d, const std::string& name, u32* at_lsn, u32* at) {
        const u32 sectors = (d.size + kSectorSize - 1) / kSectorSize;
        for (u32 s = 0; s < sectors; ++s) {
            if (!Read(d.lsn + s, 1, sector.data())) return false;
            for (u32 o = 0; o < kSectorSize;) {
                const u8 length = sector[o];
                if (length == 0 || o + length > kSectorSize) break;
                const u8 n = sector[o + 32];
                const std::string got(reinterpret_cast<const char*>(&sector[o + 33]), n);
                if ((n > 1 || (n == 1 && got[0] > 1)) && Bare(got) == Bare(name)) {
                    *at_lsn = d.lsn + s;
                    *at = o;
                    return true;
                }
                o += length;
            }
        }
        return false;
    };
    // A record added to `d`: in its last sector, or - that one full - the
    // directory moved past the disc's end with a sector more.
    auto append = [&](Dir& d, const std::vector<u8>& record) {
        const u32 sectors = (d.size + kSectorSize - 1) / kSectorSize;
        u8* tail = Patch(d.lsn + sectors - 1);
        u32 used = 0;
        while (used < kSectorSize && tail[used] != 0) used += tail[used];
        if (used + record.size() <= kSectorSize) {
            std::memcpy(tail + used, record.data(), record.size());
            return;
        }
        const u32 moved = next_virtual_;
        next_virtual_ += sectors + 1;
        for (u32 s = 0; s < sectors; ++s) {
            std::vector<u8> copy(kSectorSize);
            Read(d.lsn + s, 1, copy.data());
            std::memcpy(Patch(moved + s), copy.data(), kSectorSize);
        }
        u8* fresh = Patch(moved + sectors);
        std::memset(fresh, 0, kSectorSize);
        std::memcpy(fresh, record.data(), record.size());
        d.lsn = moved;
        d.size = (sectors + 1) * kSectorSize;
        // Its own "." record, and its record in its parent (or the volume's).
        u8* self = Patch(moved);
        PutBoth32(self + 2, d.lsn);
        PutBoth32(self + 10, d.size);
        u8* parent = Patch(d.record_lsn);
        PutBoth32(parent + d.record_at + 2, d.lsn);
        PutBoth32(parent + d.record_at + 10, d.size);
    };
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        u32 at_lsn = 0, at = 0;
        if (find(dir, parts[i], &at_lsn, &at)) {
            Read(at_lsn, 1, sector.data());
            if (!(sector[at + 25] & 2)) return false;           // a file of that name
            dir = Dir{Le32(&sector[at + 2]), Le32(&sector[at + 10]), at_lsn, at};
            continue;
        }
        // A new directory: "." and "..", then its record in this one.
        const u32 made = next_virtual_++;
        u8* first = Patch(made);
        std::memset(first, 0, kSectorSize);
        const std::vector<u8> self = Record(std::string(1, char(0)), made, kSectorSize, true);
        const std::vector<u8> up = Record(std::string(1, char(1)), dir.lsn, dir.size, true);
        std::memcpy(first, self.data(), self.size());
        std::memcpy(first + self.size(), up.data(), up.size());
        append(dir, Record(parts[i], made, kSectorSize, true));
        if (!find(dir, parts[i], &at_lsn, &at)) return false;
        dir = Dir{made, kSectorSize, at_lsn, at};
    }
    // The file's data past the disc's end.
    const u32 lsn = next_virtual_;
    const u32 sectors = std::max<u32>(1, (size + kSectorSize - 1) / kSectorSize);
    next_virtual_ += sectors;
    extras_.push_back(Extra{lsn, sectors, size, host_path});
    u32 at_lsn = 0, at = 0;
    const std::string& name = parts.back();
    if (find(dir, name, &at_lsn, &at)) {
        u8* record = Patch(at_lsn);
        PutBoth32(record + at + 2, lsn);
        PutBoth32(record + at + 10, size);
        return true;
    }
    append(dir, Record(name.find(';') == std::string::npos ? name + ";1" : name, lsn, size, false));
    return true;
}

Disc& GuestDisc() {
    static Disc disc;
    return disc;
}

}  // namespace ps2host
