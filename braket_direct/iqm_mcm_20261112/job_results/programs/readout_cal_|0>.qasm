OPENQASM 3.0;
bit[10] b;
#pragma braket verbatim
box{
prx(0.0, 0.0) $1;
prx(0.0, 0.0) $2;
prx(0.0, 0.0) $3;
prx(0.0, 0.0) $4;
prx(0.0, 0.0) $5;
prx(0.0, 0.0) $6;
prx(0.0, 0.0) $7;
prx(0.0, 0.0) $10;
prx(0.0, 0.0) $11;
prx(0.0, 0.0) $12;
}
b[0] = measure $1;
b[1] = measure $2;
b[2] = measure $3;
b[3] = measure $4;
b[4] = measure $5;
b[5] = measure $6;
b[6] = measure $7;
b[7] = measure $10;
b[8] = measure $11;
b[9] = measure $12;