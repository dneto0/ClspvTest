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

kernel void greaterthan_const(__global float* outDest, int inWidth, int inHeight,
                        int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int x_cmp = x + offset;

  int index = (y * inWidth) + x;

  float value = 0.0f;
  switch (y) {
    case 0:
      value = (x_cmp > -4) ? 1.0f : -1.0f;
      break;
    case 1:
      value = (x_cmp > -3) ? 1.0f : -1.0f;
      break;
    case 2:
      value = (x_cmp > -2) ? 1.0f : -1.0f;
      break;
    case 3:
      value = (x_cmp > -1) ? 1.0f : -1.0f;
      break;
    case 4:
      value = (x_cmp >  0) ? 1.0f : -1.0f;
      break;
    case 5:
      value = (x_cmp >  1) ? 1.0f : -1.0f;
      break;
    case 6:
      value = (x_cmp >  2) ? 1.0f : -1.0f;
      break;
    case 7:
      value = (x_cmp >  3) ? 1.0f : -1.0f;
      break;
    default:
      break;
  }
  outDest[index] = value;
}

// Note: This gets compiled down to OpSignedLessThan
kernel void greaterthan_const_left(__global float* outDest, int inWidth,
                                   int inHeight, int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int x_cmp = x + offset;

  int index = (y * inWidth) + x;

  float value = 0.0f;
  switch (y) {
    case 0:
      value = (-4 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 1:
      value = (-3 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 2:
      value = (-2 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 3:
      value = (-1 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 4:
      value = (0 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 5:
      value = (1 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 6:
      value = (2 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 7:
      value = (3 > x_cmp) ? 1.0f : -1.0f;
      break;
    default:
      break;
  }
  outDest[index] = value;
}
