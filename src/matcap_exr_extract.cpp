#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfInputFile.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool starts_with(const std::string &value, const std::string &prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool has_layer_prefix(const OPENEXR_IMF_NAMESPACE::ChannelList &channels, const std::string &prefix)
{
    const std::string match = prefix + ".";
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        if (starts_with(it.name(), match)) {
            return true;
        }
    }
    return false;
}

std::vector<float> read_layer_rgba(
    OPENEXR_IMF_NAMESPACE::InputFile &file,
    const OPENEXR_IMF_NAMESPACE::ChannelList &channels,
    const IMATH_NAMESPACE::Box2i &data_window,
    int width,
    int height,
    const std::string &prefix)
{
    static constexpr const char *suffixes[4] = {"R", "G", "B", "A"};
    std::vector<float> planes[4];
    bool present[4] = {false, false, false, false};
    bool has_any_channel = false;
    const std::int64_t row_stride = std::int64_t(width) * sizeof(float);
    OPENEXR_IMF_NAMESPACE::FrameBuffer frame_buffer;

    for (int channel_index = 0; channel_index < 4; ++channel_index) {
        std::string channel_name = prefix.empty() ? suffixes[channel_index] : prefix + "." + suffixes[channel_index];
        if (channels.findChannel(channel_name.c_str()) == nullptr) {
            continue;
        }

        has_any_channel = true;
        present[channel_index] = true;
        planes[channel_index].assign(std::size_t(width) * std::size_t(height), 0.0f);
        char *base = reinterpret_cast<char *>(planes[channel_index].data()) -
                     std::ptrdiff_t(data_window.min.x) * sizeof(float) -
                     std::ptrdiff_t(data_window.min.y) * row_stride;
        frame_buffer.insert(
            channel_name.c_str(),
            OPENEXR_IMF_NAMESPACE::Slice(
                OPENEXR_IMF_NAMESPACE::FLOAT,
                base,
                sizeof(float),
                row_stride));
    }

    if (!has_any_channel) {
        throw std::runtime_error("EXR matcap layer is missing RGBA channels");
    }

    file.setFrameBuffer(frame_buffer);
    file.readPixels(data_window.min.y, data_window.max.y);

    std::vector<float> rgba(std::size_t(width) * std::size_t(height) * 4u, 0.0f);
    for (int y = 0; y < height; ++y) {
        const int source_y = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            const std::size_t src_idx = std::size_t(source_y) * std::size_t(width) + std::size_t(x);
            const std::size_t dst_idx = (std::size_t(y) * std::size_t(width) + std::size_t(x)) * 4u;
            for (int channel_index = 0; channel_index < 4; ++channel_index) {
                rgba[dst_idx + std::size_t(channel_index)] = present[channel_index] ?
                    planes[channel_index][src_idx] : (channel_index == 3 ? 1.0f : 0.0f);
            }
        }
    }

    return rgba;
}

void write_binary(const fs::path &path, const std::vector<float> &values)
{
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }
    file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("Failed to write output file: " + path.string());
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        if (argc < 3) {
            throw std::runtime_error("Usage: matcap_exr_extract <input.exr> <output_dir>");
        }

        const fs::path input_path = argv[1];
        const fs::path output_dir = argv[2];
        fs::create_directories(output_dir);

        OPENEXR_IMF_NAMESPACE::InputFile file(input_path.string().c_str());
        const OPENEXR_IMF_NAMESPACE::Header &header = file.header();
        const OPENEXR_IMF_NAMESPACE::ChannelList &channels = header.channels();
        const IMATH_NAMESPACE::Box2i &data_window = header.dataWindow();
        const int width = data_window.max.x - data_window.min.x + 1;
        const int height = data_window.max.y - data_window.min.y + 1;
        const bool has_diffuse = has_layer_prefix(channels, "diffuse");
        const bool has_specular = has_diffuse && has_layer_prefix(channels, "specular");

        const std::vector<float> diffuse = read_layer_rgba(
            file,
            channels,
            data_window,
            width,
            height,
            has_diffuse ? "diffuse" : "");
        write_binary(output_dir / "diffuse.bin", diffuse);

        if (has_specular) {
            const std::vector<float> specular = read_layer_rgba(
                file,
                channels,
                data_window,
                width,
                height,
                "specular");
            write_binary(output_dir / "specular.bin", specular);
        }

        std::cout << width << ',' << height << ',' << (has_specular ? 1 : 0) << std::endl;
        return 0;
    }
    catch (const std::exception &exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
