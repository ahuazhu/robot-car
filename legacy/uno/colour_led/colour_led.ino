int redpin=11;
int bluepin=9;
int greenpin=10;
int redval;
int blueval;
int greenval;
int i=0;
void setup()
{
  pinMode(redpin,OUTPUT);
  pinMode(bluepin,OUTPUT);
  pinMode(greenpin,OUTPUT);
  pinMode(redval,INPUT);
  pinMode(blueval,INPUT);
  pinMode(greenval,INPUT);
  Serial.begin(9600);
}
void loop()
{
  i++;
  if(i<200)
{
    redval+=1;
    greenval-=1;
    blueval=1;
}
  else if(i<400)
{
    redval-=1;
    greenval=1;
    blueval+=1;
}
else if(i<600)
{
  redval=1;
  greenval+=1;
  blueval-=1;
}
else
{
  i=0;
}
analogWrite(redpin,redval);
analogWrite(bluepin,blueval);
analogWrite(greenpin,greenval);
Serial.print(i,DEC);
Serial.print(  "  R:");
Serial.print(redval,DEC);
Serial.print(  "  B:");
Serial.print(blueval,DEC);
Serial.print(  "  G:");
Serial.print(greenval,DEC);
}
  