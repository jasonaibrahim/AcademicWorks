Feature: user can login to ronde

  As a new visitor
  So I can access my account and its settings
  I want to be able to login to ronde using different methods

Background: I am visiting the website for the first time this session

 Given the following users exist:
      | email 	     	  | first_name | last_name | phone_number | password	  |
      | ronde@gmail.com   | RondeFirst | Rondelast | 5105105101   | RondePassword | 
      | sonde@gmail.com   | SondeFirst | SondeLast | 5105105102	  | SondePassword |

 And I am on the Ronde home page
 

Scenario: Logging in by entering details

 When I follow "Sign In"

 When I fill in "ronde@gmail.com" for "Email"

 When I fill in "RondePassword" for "Password"

 When I press "Sign in"	

 Then I should see "Signed in successfully."

Scenario: Logging in with invalid password

 When I follow "Sign In"

 When I fill in "ronde@gmail.com" for "Email"

 When I fill in "electricboogaloo" for "Password"

 When I press "Sign in"

 Then I should see "Invalid email or password."

Scenario: Logging in with invalid email

 When I follow "Sign In"

 When I fill in "electricboogaloo@gmail.com" for "Email"

 When I fill in "RondePassword" for "Password"

 When I press "Sign in"

 Then I should see "Invalid email or password."

 