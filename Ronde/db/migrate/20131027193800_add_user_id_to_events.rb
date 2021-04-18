class AddUserIdToEvents < ActiveRecord::Migration
	def change
		create_table :events do |t|
			t.string :name
			t.text :attending
			t.string :location
			t.text :description
			t.boolean :public
			t.datetime :start
			t.datetime :end
			t.timestamps
			t.belongs_to :user
			t.integer :user_id
		end
	end
end
