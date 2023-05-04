#ifndef _COUT_COLOR_H_
#define _COUT_COLOR_H_

#include <iostream>
#include <string>

/* FOREGROUND */
#define COLORRST  "\x1B[0m"
#define COLORRED  "\x1B[31m"
#define COLORGRN  "\x1B[32m"
#define COLORYEL  "\x1B[33m"
#define COLORBLU  "\x1B[34m"
#define COLORMAG  "\x1B[35m"
#define COLORCYN  "\x1B[36m"
#define COLORWHT  "\x1B[37m"

#define FONTRED(x)     COLORRED x COLORRST
#define FONTGREEN(x)   COLORGRN x COLORRST
#define FONTYELLOW(x)  COLORYEL x COLORRST
#define FONTBLUE(x)    COLORBLU x COLORRST
#define FONTMAGENTA(x) COLORMAG x COLORRST
#define FONTCYAN(x)    COLORCYN x COLORRST
#define FONTWHITE(x)   COLORWHT x COLORRST

#define BOLD(x)      "\x1B[1m" x COLORRST
#define UNDERLINE(x) "\x1B[4m" x COLORRST

namespace colorcode{
	const std::string text_black   = "\033[0;30m";
	const std::string text_red     = "\033[0;31m";
	const std::string text_green   = "\033[0;32m";
	const std::string text_yellow  = "\033[0;33m";
	const std::string text_blue    = "\033[0;34m";
	const std::string text_magenta = "\033[0;35m";
	const std::string text_cyan    = "\033[0;36m";
	const std::string text_white   = "\033[0;37m";

	const std::string background_black   = "\033[0;40m";
	const std::string background_red     = "\033[0;41m";
	const std::string background_green   = "\033[0;42m";
	const std::string background_yellow  = "\033[0;43m";
	const std::string background_blue    = "\033[0;44m";
	const std::string background_magenta = "\033[0;45m";
	const std::string background_cyan    = "\033[0;46m";
	const std::string background_white   = "\033[0;47m";

	const std::string cout_reset     = "\033[0m";
	const std::string cout_bold      = "\033[1m";
	const std::string cout_underline = "\033[4m";
	const std::string cout_inverse   = "\033[7m";

	const std::string cout_boldoff      = "\033[21m";
	const std::string cout_underlineoff = "\033[24m";
	const std::string cout_inverseoff   = "\033[27m";

};

#endif