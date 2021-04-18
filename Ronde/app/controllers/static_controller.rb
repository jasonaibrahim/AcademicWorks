class StaticController < ApplicationController
	def about
	end

	def contact
	end

	def terms_of_use
	end

	def tour
	end

  def home
    if current_user && current_user.sign_in_count == 1
      current_user.sign_in_count += 1
      current_user.save
      redirect_to "/static/tour"
    end
  end
end
