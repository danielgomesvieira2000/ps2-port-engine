/* A player's own music: MP3s in a folder beside the game, named
 * "Artist - Title.mp3", for a game's port to add to its music (Burnout 3's
 * EA Trax: games/burnout3/port/custom_songs.cpp) and play through
 * ps2host/music.h.
 *
 * The names come out as a PS2 game's font can show them: ASCII, accented
 * Latin letters as their plain ones, the rest left out, spaces squeezed, cut
 * with "..." past what the game's line has room for.
 */

#ifndef PS2HOST_CUSTOM_SONGS_H
#define PS2HOST_CUSTOM_SONGS_H

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace ps2host {

struct CustomSong {
    std::filesystem::path file;
    std::u16string title, artist;
};

/* The folder: PS2HOST_CUSTOM_SONGS if set, else `name` beside the settings
   file (the game), else in the current directory. */
std::filesystem::path CustomSongsFolder(const char* name = "custom_songs");

/* `text` as the game's font shows it, at most `most` characters. */
std::u16string CustomSongText(const std::wstring& text, std::size_t most);

/* A file's song: the artist before the first " - " of its name, the title
   after (without one, the name is the title, the artist "Unknown Artist"). */
CustomSong CustomSongOf(const std::filesystem::path& file, std::size_t most_chars);

/* The folder's MP3s, by file name (any case), the first `most` of them. */
std::vector<CustomSong> ScanCustomSongs(const std::filesystem::path& folder, std::size_t most,
                                        std::size_t most_chars);

}  // namespace ps2host

#endif  // PS2HOST_CUSTOM_SONGS_H
