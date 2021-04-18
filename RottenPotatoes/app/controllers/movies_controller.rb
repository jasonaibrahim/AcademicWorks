class MoviesController < ApplicationController

  def show
    id = params[:id] # retrieve movie ID from URI route
    @movie = Movie.find(id) # look up movie by unique ID
    # will render app/views/movies/show.<extension> by default
  end

  def index
    self.index_helper
    self.filter_ratings
    self.sort_view
    if @redirect == true
      redirect_to movies_url :sort => session[:sort], :ratings => session[:ratings]
    end
  end

  def index_helper
    @rating_filter = self.get_all_ratings
    @sort_by_title = false
    @sort_by_date = false

    if params[:ratings] 
      @rating_filter = params[:ratings].keys
      session[:ratings] = params[:ratings]
    elsif session[:ratings]
      @rating_filter = session[:ratings]
      @redirect = true
    end

    if params[:sort]
      @sort_key = params[:sort]
      session[:sort] = params[:sort]
    elsif session[:sort]
      @sort_key = session[:sort]
      @redirect = true
    end 

  end

  def filter_ratings
    if @rating_filter
      @movies = Movie.all.select{ |movie| @rating_filter.include?(movie.rating)}
    else
      @movies = Movie.all
    end
  end

  def get_all_ratings
    @all_ratings = Movie.uniq.pluck(:rating)
  end

  def sort_view
    if @sort_key == 'by_title'
      @sort_by_title = true
      @movies = @movies.sort_by &:title
    elsif @sort_key == 'by_date'
      @sort_by_date = true
      @movies = @movies.sort_by &:release_date
    else
      @movies
    end
  end

  def new
    # default: render 'new' template
  end

  def create
    @movie = Movie.create!(params[:movie])
    flash[:notice] = "#{@movie.title} was successfully created."
    redirect_to movies_path
  end

  def edit
    @movie = Movie.find params[:id]
  end

  def update
    @movie = Movie.find params[:id]
    @movie.update_attributes!(params[:movie])
    flash[:notice] = "#{@movie.title} was successfully updated."
    redirect_to movie_path(@movie)
  end

  def destroy
    @movie = Movie.find(params[:id])
    @movie.destroy
    flash[:notice] = "Movie '#{@movie.title}' deleted."
    redirect_to movies_path
  end

end
