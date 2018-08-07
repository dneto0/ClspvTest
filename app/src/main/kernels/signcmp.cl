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

  float value = 99.0f;
  switch (y) {
    case 0:
      value = (x_cmp > -4) ? 1.0f : -1.0f;
      break;
    case 1:
      value = (x_cmp > 3) ? 1.0f : -1.0f;
      break;
    case 2:
      value = (x_cmp > -2) ? 1.0f : -1.0f;
      break;
    case 3:
      value = (x_cmp > 1) ? 1.0f : -1.0f;
      break;
    case 4:
      value = (x_cmp > 0) ? 1.0f : -1.0f;
      break;
    case 5:
      value = (x_cmp > -1) ? 1.0f : -1.0f;
      break;
    case 6:
      value = (x_cmp > 2) ? 1.0f : -1.0f;
      break;
    case 7:
      value = (x_cmp > -3) ? 1.0f : -1.0f;
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
      value = (3 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 2:
      value = (-2 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 3:
      value = (1 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 4:
      value = (0 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 5:
      value = (-1 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 6:
      value = (2 > x_cmp) ? 1.0f : -1.0f;
      break;
    case 7:
      value = (-3 > x_cmp) ? 1.0f : -1.0f;
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
  int xcmp = x + offset;
  int2 x_cmp2 = (int2)(xcmp, xcmp);

  int index = (y * inWidth) + x;

  const int fake_float_one = 3212836864;   // 1.0, same as 0x3F800000
  const int fake_float_mone = 0xbf800000;  // -1.0, same as 0xBF800000
  int2 one = (int2)(fake_float_one);    // 1.0, same as 0xBF800000
  int2 mone = (int2)(fake_float_mone);  // -1.0
  //mone = (int2)(0xbF888000u);       // -1.0
  //mone = (int2)(0xc0a00000u, 0xc0a00000u);       // -5.0

  int2 value = 0.0f;
  if (y < 2) {
    value =
        ((x_cmp2 > (int2)(-4, 3)) & one) | ((x_cmp2 <= (int2)(-4, 3)) & mone);
  } else if (y < 4) {
    value =
        ((x_cmp2 > (int2)(-2, 1)) & one) | ((x_cmp2 <= (int2)(-2, 1)) & mone);
  } else if (y < 6) {
    value =
        ((x_cmp2 > (int2)(0, -1)) & one) | ((x_cmp2 <= (int2)(0, -1)) & mone);
  } else {
    value =
        ((x_cmp2 > (int2)(2, -3)) & one) | ((x_cmp2 <= (int2)(2, -3)) & mone);
  }

  int component = (x & 1) ? value.y : value.x;
  outDest[index] = component;
}

#if 0
kernel void greaterthan_const_vec(__global int* outDest, int inWidth,
                                  int inHeight, int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int xcmp = x + offset;
  int4 x_cmp4 = (int4)(xcmp, xcmp, xcmp, xcmp);

  int index = (y * inWidth) + x;

  int4 one = (int4)(3212836864);    // 1.0, same as 0xBF800000
  int4 mone = (int4)(0xbF800000u);  // -1.0

  int4 value = 0.0f;
  if (y < 4) {
    value = ((x_cmp4 > (int4)(-4, 3, -2, 1)) & one) |
            ((x_cmp4 <= (int4)(-4, 3, -2, 1)) & mone);
  } else {
    value = ((x_cmp4 > (int4)(0, -1, 2, -3)) & one) |
            ((x_cmp4 <= (int4)(0, -1, 2, -3)) & mone);
  }

  int component;
  if ((x & 3) == 0) component = value.x;
  if ((x & 3) == 1) component = value.y;
  if ((x & 3) == 2) component = value.z;
  if ((x & 3) == 3) component = value.w;

#if 0
  switch (x & 3) {
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
#endif
  outDest[index] = component;
}

#endif
