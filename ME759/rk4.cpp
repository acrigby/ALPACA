#include <stdio.h>
#include <iostream>
#include <random>



using namespace std;

int main(int argc, char *argv[]) {

    float theta1 = atoi(argv[1]);
    float theta2 = atoi(argv[2]);
    float dtheta1 = atoi(argv[3]);
    float dtheta2 = atoi(argv[4]);
    float a = atoi(argv[5]);

    float dt = 0.2;
    float pi = 3.14158 ; 
    float g = 9.81;

    float l1 = 1.0;  // [m]
    float l2 = 1.0;  // [m]
    float m1 = 1.0;  //: [kg] mass of link 1
    float m2 = 1.0 ; //: [kg] mass of link 2
    float lc1 = 0.5 ; //: [m] position of the center of mass of link 1
    float lc2 = 0.5 ; //: [m] position of the center of mass of link 2
    float I1 = 1.0 ; //: moments of inertia for both links
    float I2 = 1.0 ; //: moments of inertia for both links

    float d1 = (
            m1 * lc1*lc1
            + m2 * (l1*l1 + lc2*lc2 + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
        );
    float d2 = m2 * (lc2*lc2 + l1 * lc2 * cos(theta2)) + I2 ;
    float phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0);
    float phi1 = (
            -m2 * l1 * lc2 * dtheta2*dtheta2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        );
    float ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1*dtheta1 * sin(theta2) - phi2
            ) / (m2 * lc2*lc2 + I2 - d2*d2 / d1);
    float ddtheta1 = -(d2 * ddtheta2 + phi1) / d1 ;

    cout << dtheta1 <<','<<dtheta2 << ','<< ddtheta1<< ','<< ddtheta2<< ','<< 0.0;
    cout.flush();

    return 0;

}