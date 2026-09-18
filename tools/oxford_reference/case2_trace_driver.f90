program case2_trace_driver
  use iso_fortran_env, only: int32, real64
  use case2_reference_module, only: case2_history
  implicit none
  integer :: unit, argument_status
  character(len=1024) :: output_path

  call get_command_argument(1, output_path, status=argument_status)
  if (argument_status /= 0 .or. len_trim(output_path) == 0) then
    error stop "usage: case2_trace_driver OUTPUT"
  end if
  open(newunit=unit, file=trim(output_path), access="stream", &
       form="unformatted", status="replace", action="write")
  write(unit) int(5, int32)
  call write_linear(unit, 1, 1, 1, 0.005_real64, 50)
  call write_linear(unit, 2, 1, 1, -0.005_real64, 50)
  call write_linear(unit, 3, 1, 2, 0.01_real64, 50)
  call write_piecewise(unit, 4, [0.004_real64, 0.0_real64], 30)
  call write_piecewise(unit, 5, [0.003_real64, 0.0_real64, 0.005_real64], 25)
  close(unit)

contains

  subroutine write_linear(unit_number, history_id, row, column, final_value, steps)
    integer, intent(in) :: unit_number, history_id, row, column, steps
    real(real64), intent(in) :: final_value
    real(real64), allocatable :: history(:, :, :)
    integer :: step

    allocate(history(3, 3, steps))
    do step = 1, steps
      call identity(history(:, :, step))
      history(row, column, step) = history(row, column, step) + &
          final_value * real(step, real64) / real(steps, real64)
    end do
    call solve_and_write(unit_number, history_id, history)
  end subroutine write_linear


  subroutine write_piecewise(unit_number, history_id, targets, steps_per_segment)
    integer, intent(in) :: unit_number, history_id, steps_per_segment
    real(real64), intent(in) :: targets(:)
    real(real64), allocatable :: history(:, :, :)
    real(real64) :: start_value, value
    integer :: segment, local_step, step, total_steps

    total_steps = size(targets) * steps_per_segment
    allocate(history(3, 3, total_steps))
    start_value = 0.0_real64
    step = 0
    do segment = 1, size(targets)
      do local_step = 1, steps_per_segment
        step = step + 1
        value = start_value + (targets(segment) - start_value) * &
            real(local_step, real64) / real(steps_per_segment, real64)
        call identity(history(:, :, step))
        history(1, 1, step) = history(1, 1, step) + value
      end do
      start_value = targets(segment)
    end do
    call solve_and_write(unit_number, history_id, history)
  end subroutine write_piecewise


  subroutine solve_and_write(unit_number, history_id, history)
    integer, intent(in) :: unit_number, history_id
    real(real64), intent(in) :: history(:, :, :)
    real(real64), allocatable :: dt(:), stress(:, :), fp(:, :, :)
    real(real64), allocatable :: orientation(:, :, :), slip(:, :), ssd(:, :)
    real(real64), allocatable :: crss(:, :)
    integer(int32), allocatable :: status32(:)
    integer, allocatable :: status(:)
    integer :: steps

    steps = size(history, 3)
    allocate(dt(steps), stress(6, steps), fp(3, 3, steps))
    allocate(orientation(3, 3, steps), slip(12, steps), ssd(12, steps))
    allocate(crss(12, steps), status(steps), status32(steps))
    dt = 1.0e-8_real64
    call case2_history(history, dt, 293.15_real64, steps, stress, fp, &
        orientation, slip, ssd, crss, status)
    status32 = int(status, int32)
    write(unit_number) int(history_id, int32), int(steps, int32)
    write(unit_number) history
    write(unit_number) dt
    write(unit_number) stress
    write(unit_number) fp
    write(unit_number) orientation
    write(unit_number) slip
    write(unit_number) ssd
    write(unit_number) crss
    write(unit_number) status32
  end subroutine solve_and_write


  subroutine identity(matrix)
    real(real64), intent(out) :: matrix(3, 3)
    matrix = 0.0_real64
    matrix(1, 1) = 1.0_real64
    matrix(2, 2) = 1.0_real64
    matrix(3, 3) = 1.0_real64
  end subroutine identity

end program case2_trace_driver
