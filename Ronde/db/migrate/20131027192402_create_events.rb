class CreateEvents < ActiveRecord::Migration
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
    end
  end
end
