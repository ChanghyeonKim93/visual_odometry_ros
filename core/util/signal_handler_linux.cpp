#include "core/util/signal_handler_linux.h"

namespace signal_handler{
    void callbackSignal(sig_atomic_t sig){
        printf("::::::::::::::: Received SIGINT(%d) :::::::::::::::\n", sig);
        // signal(sig, SIG_IGN);
        throw std::runtime_error("user SIGINT is received.");
    };

    void initSignalHandler(){
        signal(SIGINT, callbackSignal);
    };
};
