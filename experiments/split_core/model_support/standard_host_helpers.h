#pragma once
template<class T> std::vector<T> ReadFixture(const std::string& name) {
    const char* fixtureRoot=std::getenv("KVIP_FIXTURE_DIR");
    if(!fixtureRoot)throw std::runtime_error("Set KVIP_FIXTURE_DIR to the prepared Model fixture");
    std::ifstream file(std::string(fixtureRoot) + "/" + name, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("missing fixture " + name);
    auto bytes = file.tellg();
    if (bytes < 0 || uint64_t(bytes) % sizeof(T)) throw std::runtime_error("fixture alignment");
    std::vector<T> result(uint64_t(bytes) / sizeof(T));
    file.seekg(0); file.read(reinterpret_cast<char*>(result.data()), bytes);
    if (!file) throw std::runtime_error("short fixture read");
    return result;
}
uint64_t ExpectedPayload(uint32_t pool, uint64_t token, uint32_t field) {
    if (field < 64) {
        constexpr uint8_t code[7] = {0xC4, 0xC0, 0xB8, 0, 0x38, 0x40, 0x44};
        uint64_t value = 0;
        for (uint32_t j = 0; j < 8; ++j) {
            uint64_t column = uint64_t(field) * 8 + j;
            value |= uint64_t(code[((token >> (field % 18)) + column + uint64_t(pool) * 3) % 7]) << (8 * j);
        }
        return value;
    }
    if (field < 80) {
        uint64_t value = 0;
        for (uint32_t j = 0; j < 4; ++j) {
            uint32_t column = (field - 64) * 4 + j;
            uint16_t word = uint16_t(0x3E80U | (((token >> ((column / 4) % 18)) + column + uint64_t(pool) * 11) & 127U));
            value |= uint64_t(word) << (16 * j);
        }
        return value;
    }
    if (field == 80) return UINT64_C(0x3D8000003D000000);
    if (field == 81) return UINT64_C(0x3E8000003E000000);
    throw std::out_of_range("payload field >=82");
}
