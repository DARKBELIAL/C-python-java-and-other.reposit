#include <stdio.h>

int main() {
    char a;
    int b;
    float s;
  
    int c;

    printf("Enter a character: ");
    scanf(" %c", &a);

    printf("Enter an integer: ");
    scanf("%d", &b);

    printf("Enter a floating-point number: ");
    scanf("%f", &s);
    printf("Enter the  number whose address to be defined: ");
    scanf("%d", &c);
     

  
    
    
   

    printf("Character: %c\n", a);
    printf("Integer: %d\n",b);
    printf("Float: %f\n",s);
     printf("address: %p\n", &c);


    return 0;
}

