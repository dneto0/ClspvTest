kernel void greaterthan(__global float* outDest, int inWidth, int inHeight,
                        int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int x_cmp = x + offset;
  int y_cmp = y + offset;

  int index = (y * inWidth) + x;

  if (x < inWidth && y < inWidth) {
    outDest[index] = (x_cmp > y_cmp) ? 1.0f : -1.0f;
  } else {
    outDest[index] = 0.0f;
  }
}
