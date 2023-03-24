#include "wrapper/ros1/util/ros_print_in_color.h"

namespace pc
{
   std::ostream& operator<<(std::ostream& os, PRINT_COLOR c)
  {
    switch(c)
    {
      case BLACK    : os << "\033[1;30m"; break;
      case RED      : os << "\033[1;31m"; break;
      case GREEN    : os << "\033[1;32m"; break;
      case YELLOW   : os << "\033[1;33m"; break;
      case BLUE     : os << "\033[1;34m"; break;
      case MAGENTA  : os << "\033[1;35m"; break;
      case CYAN     : os << "\033[1;36m"; break;
      case WHITE    : os << "\033[1;37m"; break;
      case ENDCOLOR : os << "\033[0m";    break;
      default       : os << "\033[1;37m";
    }
    return os;
  };
 //namespace pc
}