#################################
## Rain alert message
#################################

uint8 NO_RAIN=0
uint8 WARNING=1
uint8 ALERT=2
uint8 RAIN=3

Header      header
int32       time
int8        status10m
int8        status20m
int8        status30m
int8        status40m
int8        status50m
int8        status60m
