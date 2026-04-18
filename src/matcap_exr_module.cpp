#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfInputFile.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace {

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool has_layer_prefix(const OPENEXR_IMF_NAMESPACE::ChannelList& channels, const std::string& prefix)
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
    OPENEXR_IMF_NAMESPACE::InputFile& file,
    const OPENEXR_IMF_NAMESPACE::ChannelList& channels,
    const IMATH_NAMESPACE::Box2i& data_window,
    int width,
    int height,
    const std::string& prefix)
{
    static constexpr const char* suffixes[4] = {"R", "G", "B", "A"};
    std::vector<float> planes[4];
    bool present[4] = {false, false, false, false};
    bool has_any_channel = false;
    const std::int64_t row_stride = std::int64_t(width) * sizeof(float);
    OPENEXR_IMF_NAMESPACE::FrameBuffer frame_buffer;

    for (int channel_index = 0; channel_index < 4; ++channel_index) {
        const std::string channel_name = prefix.empty() ? suffixes[channel_index] : prefix + "." + suffixes[channel_index];
        if (channels.findChannel(channel_name.c_str()) == nullptr) {
            continue;
        }

        has_any_channel = true;
        present[channel_index] = true;
        planes[channel_index].assign(std::size_t(width) * std::size_t(height), 0.0f);
        char* base = reinterpret_cast<char*>(planes[channel_index].data()) -
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
                rgba[dst_idx + std::size_t(channel_index)] = present[channel_index]
                    ? planes[channel_index][src_idx]
                    : (channel_index == 3 ? 1.0f : 0.0f);
            }
        }
    }

    return rgba;
}

nb::dict extract_matcap_exr_impl(const std::string& path)
{
    OPENEXR_IMF_NAMESPACE::InputFile file(path.c_str());
    const OPENEXR_IMF_NAMESPACE::Header& header = file.header();
    const OPENEXR_IMF_NAMESPACE::ChannelList& channels = header.channels();
    const IMATH_NAMESPACE::Box2i& data_window = header.dataWindow();
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

    std::vector<float> specular;
    if (has_specular) {
        specular = read_layer_rgba(file, channels, data_window, width, height, "specular");
    }

    nb::dict result;
    result["width"] = width;
    result["height"] = height;
    result["has_specular"] = has_specular;
    result["diffuse"] = nb::bytes(reinterpret_cast<const char*>(diffuse.data()), diffuse.size() * sizeof(float));
    result["specular"] = has_specular
        ? nb::bytes(reinterpret_cast<const char*>(specular.data()), specular.size() * sizeof(float))
        : nb::bytes("", 0);
    return result;
}

}  // namespace

NB_MODULE(mathops_matcap_native, m)
{
    m.doc() = "MathOPS EXR matcap loader";
    m.def("extract_matcap_exr", &extract_matcap_exr_impl, nb::arg("path"));
}
