#ifndef _COUT_COLOR_H_
#define _COUT_COLOR_H_

#include <iostream>
#include <string>

namespace colorcode
{
	const std::string text_black   = "\033[0;30m";
	const std::string text_red     = "\033[0;31m";
	const std::string text_green   = "\033[0;32m";
	const std::string text_yellow  = "\033[0;33m";
	const std::string text_blue    = "\033[0;34m";
	const std::string text_magenta = "\033[0;35m";
	const std::string text_cyan    = "\033[0;36m";
	const std::string text_white   = "\033[0;37m";

	const std::string cout_reset     = "\033[0m";
	const std::string cout_bold      = "\033[1m";
	const std::string cout_underline = "\033[4m";
	const std::string cout_inverse   = "\033[7m";

	const std::string cout_boldoff      = "\033[21m";
	const std::string cout_underlineoff = "\033[24m";
	const std::string cout_inverseoff   = "\033[27m";

};
#endif