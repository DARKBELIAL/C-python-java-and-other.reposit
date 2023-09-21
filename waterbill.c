#include <stdio.h>
#include <math.h>
# define w_demand_charge 35
//all charges are taken in dollars 
int main (){
w_demand_charge = 35;
water_demand=1.10//water demand per thousand gallons
bool isQuarterCompleted = false;
int currentmonth;
printf("Enter the current month\n");
scanf("%d",&currentmonth);
    if (currentMonth >= 1 && currentMonth <= 3) {
        isQuarterCompleted = false; 
    } else if (currentMonth >= 4 && currentMonth <= 6) {
        isQuarterCompleted = false; 
    } else if (currentMonth >= 7 && currentMonth <= 9) {
        isQuarterCompleted = false; 
    } else if (currentMonth >= 10 && currentMonth <= 12) {
        isQuarterCompleted = true; 
    } else {
        printf("Invalid input. Please enter a valid month (1-12).\n");
        return 1; 
    }

    if (isQuarterCompleted) {
    float current_reading,previous_reading;
    printf("Enter the current reading\n");
scanf("%f",&current_reading);
printf("Enter the previous reading\n");
scanf("%f",&previous_reading);
if (current_reading>previous_reading)
    float used = current_reading-previous_reading;
     float used_charge =used*water_demand;
     float u;
     printf("Enter the unpaid balance\n");
     scanf("%f",&u);
     switch(unpaid_balance);
     
     case 0:
     printf("No net charged assessed!\n");
     break;
     default:
     float water_bill=w_demand_charge +used_charge+2;
     printf("The net water bill is %f,  water_bill);
     }
return 0;
}




