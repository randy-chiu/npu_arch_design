#include "soc_v0_addr.h"

    .section .text.start
    .globl _start
_start:
    li sp, SOC_CPU_STACK_POINTER
    call main

1:
    j 1b
