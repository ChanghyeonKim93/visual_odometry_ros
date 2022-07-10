#include "util/signal_handler_linux.h"

void SignalHandle::callbackSignal(sig_atomic_t sig){
    printf("::::::::::::::: Received SIGINT(%d) :::::::::::::::\n", sig);
    // signal(sig, SIG_IGN);
    throw std::runtime_error("user SIGINT is received.");
};

void SignalHandle::initSignalHandler(){
    signal(SIGINT, SignalHandle::callbackSignal);
};