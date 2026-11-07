#pragma once
namespace rotating {
constexpr uint32_t kPostGroups=(SIM_BATCH+kServiceLanes-1)/kServiceLanes;
constexpr uint32_t kPostThreads=kPostGroups<=1?512:(kPostGroups<=2?256:(kPostGroups<=4?128:64));
constexpr uint32_t kPostBank=kPostGroups<=1?448:(kPostGroups<=2?240:(kPostGroups<=4?120:56));
}
