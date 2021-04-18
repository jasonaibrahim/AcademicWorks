require 'spec_helper'

describe User do
  describe "Creating a new User" do
  	before :each do
  		@u = User.new(
  			{:first_name => "Jason", 
  			:last_name => "Ibrahim", 
  			:email => "jason@example.com",
  			:password => "funnel cakes", 
  			:password_confirmation => "funnel cakes",
  			:phone_number => "5556789090"
  			})
  	end
  	it "should allow the creation of a valid user" do
  		@u.save
  		@u.should be_valid
  	end
  	it "should only accept properly formatted email addresses", :pending => true do
  	end
  	it "should require a first name", :pending => true do
  	end
  	it "should require a non empty first name", :pending => true do
  	end
  	it "should require a last name", :pending => true do
  	end
  	it "should require a non empty last name", :pending => true do
  	end
  	it "should require a properly formatted email", :pending => true do
  	end
  	it "should require a non-empty password", :pending => true do
  	end
  	it "should require a valid password", :pending => true do
  	end
  	it "should ask for a password confirmation", :pending => true do
  	end
  	it "should require that the password confirmation is the same as the password", :pending => true do
  	end
  	it "should require a properly formatted phone number", :pending => true do
  	end
  	it "should only save a user with a proper email, phone number, first name, last name", :pending => true do
  	end
  	it "should save the user in the database", :pending => true do
  	end
  end
end
