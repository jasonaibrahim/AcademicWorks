(define (make-tree entry children)
  (cons entry children))

(define (entry t)
  (car entry))

(define (children t)
  (cdr children))

;; utility functions
(define (leaf? t)
  (null? (children t)))
