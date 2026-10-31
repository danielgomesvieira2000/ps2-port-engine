/* A player's own music (ps2host/custom_songs.h). */

#include "ps2host/custom_songs.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cwctype>

#include "ps2host/settings.h"

namespace ps2host {

namespace fs = std::filesystem;

namespace {
/* A character the game's font has (ASCII), accented Latin letters as their
   plain ones; others dropped. */
char16_t Plain(wchar_t ch) {
    if (ch >= 0x20 && ch < 0x7F) return char16_t(ch);
    static const struct { wchar_t from, to; } kFold[] = {
        {L'À', 'A'}, {L'Á', 'A'}, {L'Â', 'A'}, {L'Ã', 'A'}, {L'Ä', 'A'}, {L'Å', 'A'},
        {L'Ç', 'C'}, {L'È', 'E'}, {L'É', 'E'}, {L'Ê', 'E'}, {L'Ë', 'E'}, {L'Ì', 'I'},
        {L'Í', 'I'}, {L'Î', 'I'}, {L'Ï', 'I'}, {L'Ñ', 'N'}, {L'Ò', 'O'}, {L'Ó', 'O'},
        {L'Ô', 'O'}, {L'Õ', 'O'}, {L'Ö', 'O'}, {L'Ø', 'O'}, {L'Ù', 'U'}, {L'Ú', 'U'},
        {L'Û', 'U'}, {L'Ü', 'U'}, {L'Ý', 'Y'}, {L'ß', 's'}, {L'à', 'a'}, {L'á', 'a'},
        {L'â', 'a'}, {L'ã', 'a'}, {L'ä', 'a'}, {L'å', 'a'}, {L'ç', 'c'}, {L'è', 'e'},
        {L'é', 'e'}, {L'ê', 'e'}, {L'ë', 'e'}, {L'ì', 'i'}, {L'í', 'i'}, {L'î', 'i'},
        {L'ï', 'i'}, {L'ñ', 'n'}, {L'ò', 'o'}, {L'ó', 'o'}, {L'ô', 'o'}, {L'õ', 'o'},
        {L'ö', 'o'}, {L'ø', 'o'}, {L'ù', 'u'}, {L'ú', 'u'}, {L'û', 'u'}, {L'ü', 'u'},
        {L'ý', 'y'}, {L'ÿ', 'y'}, {L'‘', '\''}, {L'’', '\''}, {L'“', '"'}, {L'”', '"'},
        {L'–', '-'}, {L'—', '-'}};
    for (const auto& f : kFold)
        if (f.from == ch) return char16_t(f.to);
    return 0;
}

std::wstring Lower(std::wstring text) {
    for (wchar_t& ch : text) ch = wchar_t(std::towlower(ch));
    return text;
}
}  // namespace

fs::path CustomSongsFolder(const char* name) {
    if (const char* dir = std::getenv("PS2HOST_CUSTOM_SONGS")) return fs::path(dir);
    const std::string settings = SettingsPath();
    return (settings.empty() ? fs::current_path() : fs::path(settings).parent_path()) / name;
}

std::u16string CustomSongText(const std::wstring& text, std::size_t most) {
    std::u16string out;
    for (wchar_t ch : text)
        if (char16_t p = Plain(ch)) out += p;
    // Spaces at either end, and runs of them, as one.
    std::u16string squeezed;
    for (char16_t ch : out)
        if (!(ch == u' ' && (squeezed.empty() || squeezed.back() == u' '))) squeezed += ch;
    while (!squeezed.empty() && squeezed.back() == u' ') squeezed.pop_back();
    if (most >= 3 && squeezed.size() > most) {
        squeezed.resize(most - 3);
        while (!squeezed.empty() && squeezed.back() == u' ') squeezed.pop_back();
        squeezed += u"...";
    }
    return squeezed;
}

CustomSong CustomSongOf(const fs::path& file, std::size_t most_chars) {
    const std::wstring stem = file.stem().wstring();
    CustomSong s;
    s.file = file;
    const std::size_t dash = stem.find(L" - ");
    if (dash == std::wstring::npos) {
        s.title = CustomSongText(stem, most_chars);
        s.artist = u"Unknown Artist";
    } else {
        s.artist = CustomSongText(stem.substr(0, dash), most_chars);
        s.title = CustomSongText(stem.substr(dash + 3), most_chars);
        if (s.artist.empty()) s.artist = u"Unknown Artist";
    }
    if (s.title.empty()) s.title = u"Untitled";
    return s;
}

std::vector<CustomSong> ScanCustomSongs(const fs::path& folder, std::size_t most, std::size_t most_chars) {
    std::vector<CustomSong> songs;
    std::error_code error;
    if (!fs::is_directory(folder, error)) return songs;
    for (const auto& item : fs::directory_iterator(folder, error)) {
        if (!item.is_regular_file(error)) continue;
        if (Lower(item.path().extension().wstring()) == L".mp3") songs.push_back(CustomSongOf(item.path(), most_chars));
    }
    std::sort(songs.begin(), songs.end(), [](const CustomSong& a, const CustomSong& b) {
        return Lower(a.file.filename().wstring()) < Lower(b.file.filename().wstring());
    });
    if (songs.size() > most) {
        std::fprintf(stderr, "[custom songs] %zu MP3s: the first %zu by name are played\n", songs.size(), most);
        songs.resize(most);
    }
    return songs;
}

}  // namespace ps2host
