# Communication guide

Communication metrics are intentionally layered: packet delivery, symbol
recovery, bit recovery, reconstruction distortion, prediction, task-law
preservation, and clinical utility are separate fields. `reconstruct()` only
reassembles a complete surviving packet sequence. Erasures require a decoder;
they are never silently replaced by zero.

`RLNCEncoder` emits systematic plus GF(256) coded symbols. `RLNCDecoder`
performs finite-field elimination and returns source symbols only after full
rank is reached.
