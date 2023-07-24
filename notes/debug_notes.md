# debug notes

## 20230531

the device info says:
```
Maximum number of threads per multiprocessor:  2048
Maximum number of threads per block:           1024
```

the p4 GPU allows 1024 threads per block. If the number of threads per block is greater than 1024, the kernel will not launch.

error message:
```
warning: Cuda API error detected: cudaLaunchKernel returned (0x9)
```

code:
~~~
# in preprocess.cu:preprocess
dim3 block2(32, 32, 3);
dim3 grid2((dst_w + block2.x - 1) / block2.x, (dst_h + block2.y - 1) / block2.y, (3 + block2.z - 1) / block2.z);

LOG(INFO) << "blobFromImage kernel launched with "
            << grid2.x << "x" << grid2.y << "x" << grid2.z << " blocks of "
            << block2.x << "x" << block2.y << "x" << block2.z << " threads";

// TODO: fix bug here
// TODO: flip the channel order
blobFromImage<<<grid2, block2>>>(
    d_ptr_dst, (float*)this->device_ptrs[0], img_num, 
    dst_h, dst_w, 3, batch_size
);
~~~

32 * 32 * 3 = 3072 > 1024, so the kernel will not launch.

modify block2 to (16, 16, 3), the kernel will launch.

## 20230708

opencv:

When cropping a rectangle area if an images, you should use this:

~~~
cv::Mat img;
cv::Mat crop;
cv::Rect rect;

img(rect).copyTo(crop);
~~~

instead of:

~~~
crop = img(rect);
~~~

The latter would cause bugs.

## 20230723

When suddenly moving the camera, the image will be blurred. When sending the blurred image to meter-reader, random segment fault will occur.

e.g.:
- malloc(): corrupted top size
- malloc(): invalid size (unsorted)
- malloc(): unsorted double linked list corrupted
- free(): invalid next size (normal)

This bug won't occur when the camera is still.

This bug is **NOT** fixed yet.