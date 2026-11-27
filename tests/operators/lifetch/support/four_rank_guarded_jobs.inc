                // Diagnostic only. Both arms execute the original union. Failure uses its natural list.
                auto trailer = reinterpret_cast<__gm__ uint64_t*>(
                    reinterpret_cast<__gm__ uint8_t*>(ctx) + sizeof(rotating::Context));
                const uint32_t window = b * 4 + epoch;
                const uint32_t offset = window * 256;
                bool useFrozen = false;
                if (rt_rounds == 4 && epoch < 4 && count <= 256) {
                    GlobalTensor<uint32_t> observed, frozen;
                    observed.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(trailer[1]) + offset);
                    frozen.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(trailer[0]) + offset);
                    const uint32_t padded = (count + 7) / 8 * 8;
                    DataCopy(observed, ids, padded);
                    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                    // The host verified that each recorded source is unique and within the row.
                    const uint32_t expectedCount = reinterpret_cast<__gm__ uint32_t*>(trailer[2])[window];
                    if (count == expectedCount && count == candidateCount && !sparseWindow) {
                        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                        DataCopy(input, frozen, padded);
                        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                        useFrozen = true;
                        for (uint32_t j = 0; j < count; ++j) {
                            const uint32_t token = input.GetValue(j);
                            if (token >= rt_history || token / 32 >= windowWords ||
                                !(scratch.GetValue(token / 32) & (1U << (token % 32)))) {
                                useFrozen = false;
                                break;
                            }
                        }
                    }
                    // Status uses a cache line per window and is checked after stream completion.
                    auto status = reinterpret_cast<__gm__ uint32_t*>(trailer[3]) + window * 16;
                    status[0] = useFrozen ? 1U : 2U;
                    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), 64);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                if (useFrozen) {
                    DataCopy(jobs,input,((count+7)/8)*8);
                } else {
                    DataCopy(jobs,ids,((count+7)/8)*8);
                }
