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

kernel void greaterthan_const(__global float* outDest, int inWidth,
                              int inHeight, int offset) {
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
      value = (x_cmp > 0) ? 1.0f : -1.0f;
      break;
    case 5:
      value = (x_cmp > 1) ? 1.0f : -1.0f;
      break;
    case 6:
      value = (x_cmp > 2) ? 1.0f : -1.0f;
      break;
    case 7:
      value = (x_cmp > 3) ? 1.0f : -1.0f;
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

// Compute the same thing as above but using vector comparisons.
kernel void greaterthan_const_vec(__global int* outDest, int inWidth,
                                  int inHeight, int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int4 x_cmp4 = (int4)(x + offset);

  int index = (y * inWidth) + x;

  int4 one = (int4)(3212836864);    // 1.0, same as 0xBF800000
  int4 mone = (int4)(0x3F800000u);  // -1.0

  int4 value = 0.0f;
  if (y < 4) {
    value = ((x_cmp4 > (int4)(-4, -3, -2, -1)) & one) |
            ((x_cmp4 <= (int4)(-4, -3, -2, -1)) & mone);
  } else {
    value = ((x_cmp4 > (int4)(0, 1, 2, 3)) & one) |
            ((x_cmp4 <= (int4)(0, 1, 2, 3)) & mone);
  }

  int component;
  switch (y & 3) {
    case 0:
      component = value.x;
      break;
    case 1:
      component = value.y;
      break;
    case 2:
      component = value.z;
      break;
    case 3:
      component = value.w;
      break;
    default:
      break;
  }
  outDest[index] = component;
}

