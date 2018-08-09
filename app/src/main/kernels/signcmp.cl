kernel void greaterthan_m2(__global float* outDest, int inWidth, int inHeight,
                           int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int x_cmp = x + offset;
  int y_cmp = y + offset;

  int index = (y * inWidth) + x;

  if (x < inWidth && y < inWidth) {
    outDest[index] = (x_cmp > -3) ? 1.0f : -1.0f;
  } else {
    outDest[index] = 0.0f;
  }
}

kernel void greaterthan(__global float* outDest, int inWidth, int inHeight,
                        int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int x_cmp = x - offset + 3;
  int y_cmp = offset - 1 - y;

  int index = (y * inWidth) + x;

  if (x < inWidth && y < inWidth) {
    outDest[index] = (x_cmp > y_cmp) ? 1.0f : -1.0f;
  } else {
    outDest[index] = 0.0f;
  }
}

kernel void lessthan(__global float* outDest, int inWidth, int inHeight,
                        int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int x_cmp = x - offset + 3;
  int y_cmp = offset - 1 - y;

  int index = (y * inWidth) + x;

  if (x < inWidth && y < inWidth) {
    outDest[index] = (x_cmp < y_cmp) ? 1.0f : -1.0f;
  } else {
    outDest[index] = 0.0f;
  }
}

kernel void greaterequal(__global float* outDest, int inWidth, int inHeight,
                        int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int x_cmp = x - offset + 3;
  int y_cmp = offset - 1 - y;

  int index = (y * inWidth) + x;

  if (x < inWidth && y < inWidth) {
    outDest[index] = (x_cmp >= y_cmp) ? 1.0f : -1.0f;
  } else {
    outDest[index] = 0.0f;
  }
}

kernel void lessequal(__global float* outDest, int inWidth, int inHeight,
                        int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int x_cmp = x - offset + 3;
  int y_cmp = offset - 1 - y;

  int index = (y * inWidth) + x;

  if (x < inWidth && y < inWidth) {
    outDest[index] = (x_cmp <= y_cmp) ? 1.0f : -1.0f;
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

// Compute the same thing as above but using vector int2 comparisons.
kernel void greaterthan_const_vec2(__global int* outDest, int inWidth,
                                  int inHeight, int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int xcmp = x + offset;
  int2 x_cmp2 = (int2)(xcmp, xcmp);

  int index = (y * inWidth) + x;

  const int fake_float_one =  0x3f800000u;   // 1.0, same as 0x3F800000
  const int fake_float_mone = 0xbf800000u;  // -1.0, same as 0xBF800000
  int2 one = (int2)(fake_float_one);
  int2 mone = (int2)(fake_float_mone);

  int2 compare_to = (int2)(0, 0);
  if (y < 2) {
    compare_to = (int2)(-4, 3);
  } else if (y < 4) {
    compare_to = (int2)(-2, 1);
  } else if (y < 6) {
    compare_to = (int2)(0, -1);
  } else if (y < 8) {
    compare_to = (int2)(2, -3);
  }
  int2 value = ((x_cmp2 > compare_to) & one) | ((x_cmp2 <= compare_to) & mone);
  int component = (y & 1) ? value.y : value.x;
  outDest[index] = component;
}

// Compute the same thing as above but using vector int4 comparisons.
kernel void greaterthan_const_vec4(__global int* outDest, int inWidth,
                                   int inHeight, int offset) {
  int x = (int)get_global_id(0);
  int y = (int)get_global_id(1);
  int xcmp = x + offset;
  int4 x_cmp4 = (int4)(xcmp);

  int index = (y * inWidth) + x;

  const int fake_float_one = 0x3f800000u;   // 1.0, same as 0x3F800000
  const int fake_float_mone = 0xbf800000u;  // -1.0, same as 0xBF800000
  int4 one = (int4)(fake_float_one);
  int4 mone = (int4)(fake_float_mone);

  int4 compare_to = (y < 4) ? (int4)(-4, 3, -2, 1) : (int4)(0, -1, 2, -3);
  int4 value = ((x_cmp4 > compare_to) & one) | ((x_cmp4 <= compare_to) & mone);
  int2 components2 = (y & 2) ? value.zw : value.xy;
  int component = (y & 1) ? components2.y : components2.x;
  outDest[index] = component;
}
