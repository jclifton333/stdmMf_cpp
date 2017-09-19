(defun njm/setup-extended-rename (number)
  (interactive)
  (find-file (format "run_infShield_500_extended_%02d_18.cpp" number))
  (write-file (format "run_infShieldState_extended_500_%02d_18.cpp" number))
  (delete-file (format "run_infShield_500_extended_%02d_18.cpp" number))
  (kill-buffer)
  (find-file (format "run_infShield_1000_extended_%02d_18.cpp" number))
  (write-file (format "run_infShieldState_extended_1000_%02d_18.cpp" number))
  (delete-file (format "run_infShield_1000_extended_%02d_18.cpp" number))
  (kill-buffer)
  )


(defun njm/setup-extended-500-file (number)
  (interactive)

  ;; open relevant file
  (find-file (format "run_infShieldState_500_extended_%02d_18.cpp" 1))

  ;; get point into position
  (beginning-of-buffer)
  (search-forward "setup networks")

  ;; comment out 100s
  (search-forward "{")
  (beginning-of-line)
  (set-mark-command nil)
  (search-forward "}")
  (comment-dwim-2)
  (deactivate-mark)

  ;; comment out 1000s
  (search-forward "{")
  (search-forward "{")
  (beginning-of-line)
  (set-mark-command nil)
  (search-forward "}")
  (comment-dwim-2)
  (deactivate-mark)

  ;; save as new file
  (write-file (format "run_infShieldState_500_extended_%02d_18.cpp" number))
  (kill-buffer)
  )

(defun njm/setup-extended-1000-file (number)
  (interactive)

  ;; open relevant file
  (find-file (format "run_infShieldState_1000_extended_%02d_18.cpp" number))

  ;; get point into position
  (beginning-of-buffer)
  (search-forward "setup networks")

  ;; comment out 100s
  (search-forward "{")
  (beginning-of-line)
  (set-mark-command nil)
  (search-forward "}")
  (comment-dwim-2)
  (deactivate-mark)

  ;; comment out 500s
  (search-forward "{")
  (beginning-of-line)
  (set-mark-command nil)
  (search-forward "}")
  (comment-dwim-2)
  (deactivate-mark)

  ;; save as new file
  (write-file (format "run_infShieldState_1000_extended_%02d_18.cpp" number))
  (kill-buffer)
  )

(defun njm/setup-extended ()
  (interactive)
  (dotimes (number 18)
    ;; (njm/setup-extended-500-file (+ number 1))
    ;; (njm/setup-extended-1000-file (+ number 1))
    (njm/setup-extended-rename (+ number 1))
    )
  )
