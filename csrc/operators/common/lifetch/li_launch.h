// Included by each transport after defining its four launch symbols.
    if (!bindings || !stream || !shape) return 100000;
    uint32_t result = LI_LAUNCH_0(shape->liCores, stream, bindings);
    if (result || !rotating::UseSequenceSplit(*shape)) return result;
    result = LI_LAUNCH_1(shape->liCores, stream, bindings);
    if (!result) result = LI_LAUNCH_2(shape->liCores, stream, bindings);
    if (!result) result = LI_LAUNCH_3(shape->liCores, stream, bindings);
    return result;
