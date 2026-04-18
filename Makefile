.PHONY: resume cv coverletter clean

CC = xelatex
OUTPUT_DIR = output
RESUME_DIR = resume
CV_DIR = cv
COVERLETTER_DIR = coverletter
RESUME_SRCS = $(shell find $(RESUME_DIR) -name '*.tex')
CV_SRCS = $(shell find $(CV_DIR) -name '*.tex')

resume : $(RESUME_DIR)/resume.tex $(RESUME_SRCS)
	$(CC) -output-directory=$(OUTPUT_DIR) $<

cv: $(CV_DIR)/cv.tex $(CV_SRCS)
	$(CC) -output-directory=$(OUTPUT_DIR) $<

coverletter : $(COVERLETTER_DIR)/coverletter.tex
	$(CC) -output-directory=$(OUTPUT_DIR) $<

clean:
	rm -rf **/*.pdf **/*.aux **/*.out **/*.log **/*.synctex.gz $(OUTPUT_DIR)/*
