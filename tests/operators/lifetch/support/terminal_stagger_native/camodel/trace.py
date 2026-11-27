# Instruments the test fixture only, immediately before the genuine SQ doorbell on both arms.
from pathlib import Path
import sys
p=Path(sys.argv[1])/"terminal_transport.h"
s=p.read_text()
old="SetFlag<HardEvent::MTE3_S>(EVENT_ID0);"
assert s.count(old)==1
p.write_text(s.replace(old,"""auto trace=reinterpret_cast<__gm__ volatile uint64_t*>(args->indices);
    auto n=trace[0];
    trace[2+2*n]=head;trace[3+2*n]=GetSystemCycle();trace[0]=n+1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(args->indices),128);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);"""))
