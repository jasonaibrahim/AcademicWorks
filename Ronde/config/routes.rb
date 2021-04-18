Ronde::Application.routes.draw do
  get "static/about"
  get "static/tour"
  get "static/home"
  get "static/terms_of_use"
  get "static/contact"

  devise_for :users, :controllers => { :omniauth_callbacks =>  "omniauth_callbacks" }

  root to: 'static#home'
end
