(let ((a 0) (b 0) (op 0))
  (write-line "Enter first number:")
  (setq a (read))

  (write-line "Enter second number:")
  (setq b (read))

  (write-line "Enter operation (+ - * /):")
  (setq op (read))

  (cond
    ((eq op '+) (print (+ a b)))
    ((eq op '-) (print (- a b)))
    ((eq op '*) (print (* a b)))
    ((eq op '/) (print (/ a b)))
    (t (print "Invalid operation"))
  )
)
