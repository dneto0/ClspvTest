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

  float fake_float_one = 5.0f;
  float fake_float_mone = -7.0f;

  float value = 99.0f;
  switch (y) {
    case 0:
      value = (x_cmp > -4) ? fake_float_one : fake_float_mone;
      break;
    case 1:
      value = (x_cmp > 3) ? fake_float_one : fake_float_mone;
      break;
    case 2:
      value = (x_cmp > -2) ? fake_float_one : fake_float_mone;
      break;
    case 3:
      value = (x_cmp > 1) ? fake_float_one : fake_float_mone;
      break;
    case 4:
      value = (x_cmp > 0) ? fake_float_one : fake_float_mone;
      break;
    case 5:
      value = (x_cmp > -1) ? fake_float_one : fake_float_mone;
      break;
    case 6:
      value = (x_cmp > 2) ? fake_float_one : fake_float_mone;
      break;
    case 7:
      value = (x_cmp > -3) ? fake_float_one : fake_float_mone;
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

  float fake_float_one = 5.0f;
  float fake_float_mone = -7.0f;

  float value = 0.0f;
  switch (y) {
    case 0:
      value = (-4 > x_cmp) ? fake_float_one : fake_float_mone;
      break;
    case 1:
      value = (3 > x_cmp) ? fake_float_one : fake_float_mone;
      break;
    case 2:
      value = (-2 > x_cmp) ? fake_float_one : fake_float_mone;
      break;
    case 3:
      value = (1 > x_cmp) ? fake_float_one : fake_float_mone;
      break;
    case 4:
      value = (0 > x_cmp) ? fake_float_one : fake_float_mone;
      break;
    case 5:
      value = (-1 > x_cmp) ? fake_float_one : fake_float_mone;
      break;
    case 6:
      value = (2 > x_cmp) ? fake_float_one : fake_float_mone;
      break;
    case 7:
      value = (-3 > x_cmp) ? fake_float_one : fake_float_mone;
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
  int2 x_cmp2 = (x & 1) ? (int2)(xcmp - 1, xcmp) : (int2)(xcmp, xcmp + 1);

  int index = (y * inWidth) + x;

  // 1.0, same as 0x3F800000
  // 5.0, same as 0x40a00000
  // -5.0, same as 0xc0a00000
  // -7.0, same as 0xc0e00000

  const int fake_float_one = 0x40a00000;
  const int fake_float_mone = 0xc0e00000;
  int2 one = (int2)(fake_float_one);
  int2 mone = (int2)(fake_float_mone);

  int2 value;
#define VAL(N) \
  value = (((x_cmp2 > (int2)(N)) & one) | ((x_cmp2 <= (int2)(N)) & mone))
  if (y == 0)
    VAL(-4);
  else if (y == 1)
    VAL(3);
  else if (y == 2)
    VAL(-2);
  else if (y == 3)
    VAL(1);
  else if (y == 4)
    VAL(0);
  else if (y == 5)
    VAL(-1);
  else if (y == 6)
    VAL(2);
  else if (y == 7)
    VAL(-3);
#undef VAL

  int component = (x & 1) ? value.y : value.x;
  outDest[index] = component;
}
