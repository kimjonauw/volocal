#import <Foundation/Foundation.h>

/// AVAudioEngine installTap/removeTap raise NSException, which Swift `catch` cannot handle.
BOOL VolocalCatchException(void (NS_NOESCAPE ^block)(void), NSError *_Nullable *_Nullable outError);

/// Bytes remaining before jetsam (`os_proc_available_memory`).
uint64_t VolocalAvailableMemory(void);
