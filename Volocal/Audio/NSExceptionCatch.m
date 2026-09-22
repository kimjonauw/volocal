#import "NSExceptionCatch.h"
#import <os/proc.h>

BOOL VolocalCatchException(void (NS_NOESCAPE ^block)(void), NSError *_Nullable *_Nullable outError) {
    @try {
        block();
        return YES;
    } @catch (NSException *exception) {
        if (outError) {
            *outError = [NSError errorWithDomain:exception.name ?: @"NSException"
                                            code:0
                                        userInfo:@{
                NSLocalizedDescriptionKey: exception.reason ?: @"AVAudioEngine exception"
            }];
        }
        return NO;
    }
}

uint64_t VolocalAvailableMemory(void) {
    return os_proc_available_memory();
}
