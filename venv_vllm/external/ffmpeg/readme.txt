FFMPEG Libraries present in this package were built using the following configure options

x64 (FFmpeg 8.1.1):   ./configure --toolchain=msvc --enable-shared --disable-static --disable-encoders --disable-decoders --enable-decoder=vp9 --disable-programs --disable-doc --disable-network
win32: ./configure --enable-shared --disable-encoders --disable-decoders --enable-decoder=vp9 --toolchain=msvc --arch=x86
x86_64: LDSOFLAGS=-Wl,-rpath,\''$$$$ORIGIN'\'  ./configure --enable-shared --disable-encoders --disable-decoders --enable-decoder=vp9  --arch=x86_64
aarch64: LDSOFLAGS=-Wl,-rpath,\''$$$$ORIGIN'\'  ./configure --enable-shared --disable-encoders --disable-decoders --enable-decoder=vp9  --arch=aarch64

Notes:
- --disable-postproc was removed as a configure option in FFmpeg 8.x and is no longer passed for the x64 build.
- The x64 build uses the canonical install layout: import .lib files and DLLs co-located in lib/x64/; matching headers in include/lib*.

Only header files and libraries that are required for compiling the sample applications inside the package have been included.



