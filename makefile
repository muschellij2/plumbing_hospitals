all: index.pdf index.html README.md index.md figures

index.pdf: index.Rmd 
	Rscript -e "rmarkdown::render('index.Rmd')"

index.html: index.Rmd 
	Rscript -e "rmarkdown::render('index.Rmd', output_format = 'html_document')"		

README.md: README.Rmd 
	Rscript -e "rmarkdown::render('README.Rmd')"

# figures: original_image.png 
figures: sketch.jpg

# original_image.png tilt_corr_image.png: make_tilted_image.R
# 	Rscript -e "source('make_tilted_image.R')"

# overlaid_slices.png: plotting_overlays.R
# 	Rscript -e "source('plotting_overlays.R')"		

index.md: index.Rmd 
	Rscript -e "rmarkdown::render('index.Rmd', output_format = rmarkdown::github_document())"

clean: 
	rm -f index.md index.pdf README.md index.html
