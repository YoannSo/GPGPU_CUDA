#ifndef __IMAGE_HPP__
#define __IMAGE_HPP__

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Image
{
  public:
	Image() = default;
    Image( const int p_width, const int p_height, const int p_nbChannels );
	~Image();

	void load( const string & p_path );
	void save( const string & p_path ) const;
	std::vector < int > getPixelRGB();
	void setPixels();
	void setRGB(std::vector<int> rgb);
  public: // All public!
	int				_width		= 0;
	int				_height		= 0;
	int				_nbChannels = 0;
	vector<int>     _pixelRGB ;
	unsigned char * _pixels		= nullptr;
};

#endif // __IMAGE_HPP__
