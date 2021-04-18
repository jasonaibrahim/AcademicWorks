class Event < ActiveRecord::Base
	belongs_to :user, class_name: 'User', foreign_key: 'user_id'
	validates_presence_of :description, :location, :name, :start
  attr_accessible :attending, :description, :end, :location, :name, :public, :start
  after_initialize :init 

  def init
  	self.public ||= true
  end
end
