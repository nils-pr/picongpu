/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Anton Helm, Rene Widera,
 *                     Richard Pausch, Alexander Debus
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace laserProfiles
        {
            namespace pulseFromSpectrum
            {
                template< typename T_Params >
                struct Unitless : public T_Params
                {
                    using Params = T_Params;

                    static constexpr float_X WAVE_LENGTH = float_X( Params::WAVE_LENGTH_SI / UNIT_LENGTH ); // unit: meter
                    static constexpr float_X v0 = (::picongpu::SI::SPEED_OF_LIGHT_SI / Params::WAVE_LENGTH_SI) * UNIT_TIME; // unit: seconds^-1
                    static constexpr float_X PULSE_LENGTH = float_X( Params::PULSE_LENGTH_SI / UNIT_TIME ); // unit: seconds (1 sigma)
                    static constexpr float_X AMPLITUDE = float_X( Params::AMPLITUDE_SI / UNIT_EFIELD ); // unit: volt / meter
                    static constexpr float_X PULSE_INIT = float_X( Params::PULSE_INIT); // unit: none
                    static constexpr float_X GDD = float_X( Params::GDD_SI / (UNIT_TIME * UNIT_TIME) ); // unit: seconds^2
                    static constexpr float_X TOD = float_X( Params::TOD_SI / (UNIT_TIME * UNIT_TIME * UNIT_TIME) ); // unit: seconds^3
                    static constexpr float_X W0 = float_X( Params::W0_SI / UNIT_LENGTH ); // unit: meter
                    static constexpr float_X FOCUS_POS = float_X( Params::FOCUS_POS_SI / UNIT_LENGTH ); // unit: meter
                    static constexpr float_X INIT_TIME = float_X( ( Params::PULSE_INIT * Params::PULSE_LENGTH_SI) / UNIT_TIME ); // unit: seconds (full inizialisation length)

                    /* Rayleigh-Length in y-direction
                     */
                    static constexpr float_X Y_R = float_X( PI * v0 * W0 * W0 / SPEED_OF_LIGHT ); // unit: meter

                    /* initialize the laser not in the first cell is equal to a negative shift
                     * in time
                     */
                    static constexpr float_X laserTimeShift = Params::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;
                    };
            } // namespace pulseFromSpectrum

            namespace acc
            {
                template<typename T_Unitless>
                struct PulseFromSpectrum : public T_Unitless
                {
                    using Unitless = T_Unitless;

                    float3_X m_elong;
                    float_X m_phase;
                    float_X m_currentStep;
                    typename FieldE::DataBoxType m_dataBoxE;
                    DataSpace<simDim> m_offsetToTotalDomain;
                    DataSpace<simDim> m_superCellToLocalOriginCellOffset;

                    /** waist of the laser beam depending on y-coordinate
                     * @param y y-coordinate
                     */
                    HDINLINE float_X Waist(float_X y)
                    {
                        // shift from PIConGPU-coordinates to coordinate system used in the equations
                        float_X y_shift = y - Unitless::FOCUS_POS;
                        return Unitless::W0 * math::sqrt( 1.0 + math::pow(y_shift / Unitless::Y_R, 2.0) );
                    }

                    /** inverse radius of curvature depending on y-coordinate
                     * @param y y-coordinate
                     */
                    HDINLINE float_X R_inv(float_X y)
                    {
                        // shift from PIConGPU-coordinates to coordinate system used in the equations
                        float_X y_shift = y - Unitless::FOCUS_POS;
                        return y_shift / ( math::pow(y_shift, 2.0) + math::pow(Unitless::Y_R, 2.0) );
                    }

                    /** gauss spectrum
                     * norm is choosen so, that the maximum has value 1
                     * v0 is the central frequency
                     * PULSE_LENGTH is used as standard sigma of gauss in time domain
                     * @param v frequency
                     */
                    HDINLINE float_X gauss_spectrum(float_X v)
                    {
                        float_X norm = 0.5 * math::sqrt(float_X(PI)) * Unitless::PULSE_LENGTH;
                        return norm * math::exp( -1.0 * math::pow( Unitless::PULSE_LENGTH * float_X(PI) * ( v - Unitless::v0 ), 2.0) );
                    }

                    /** transversal spectrum at the init plane
                     * to implement a laser-pulse with a transversal profile, the spectrum has to be depending on the distance to the beam axis:
                     * spectrum(v) --> spectrum(v, r) (the transversal profile is radial symmetric)
                     * equation: spectrum_transversal(v, r) = spectrum(v) * (1 + Focus_Pos^2 / Y_R^2 )^-1/2 * exp( -r^2 / Waist^2 )
                     * with spectrum(v) as the spectrum one would have without a transversal profile (here a gaussian spectrum)
                     * @param v frequency
                     * @param r2 distance to beam axis to the power of 2
                     */
                    HDINLINE float_X transversal_spectrum(float_X v, float_X r2)
                    {
                        float_X a = 1.0 + math::pow(Unitless::FOCUS_POS / Unitless::Y_R, 2.0);
                        float_X transversal_spectrum = gauss_spectrum(v) * math::pow(a, -0.5) * math::exp( -1.0*r2 / math::pow(Waist(0.0), 2.0));
                        return transversal_spectrum;
                    }

                    /** phase of the pulse depending on frequency and radius to beam axis at the init plane
                     * the phase is altered for two purposes:
                     * 1. to be able to choose freely the GDD/TOD of the laser-pulse
                     * 2. to implement the transversal profile (thus the phase depends on the distance to the beam axis)
                     * @param v frequency
                     * @param r2 distance to beam axis to the power of 2
                     */
                    HDINLINE float_X phase_v(float_X v, float_X r2)
                    {
                        float_X phase_shift_GDD = float_X( 0.5 * Unitless::GDD * math::pow( 2.0 * float_X(PI) * ( v - Unitless::v0 ), 2.0) );
                        float_X phase_shift_TOD = float_X( (1.0/6.0) * Unitless::TOD * math::pow( 2.0 * float_X(PI) * (v - Unitless::v0), 3.0) );
                        float_X phase_shift_transversal_1 = float_X( -1.0 * math::atan( -1.0*Unitless::FOCUS_POS / Unitless::Y_R ) );
                        float_X phase_shift_transversal_2 = float_X( -2.0 * PI * v * Unitless::FOCUS_POS / SPEED_OF_LIGHT );
                        float_X phase_shift_transversal_3 = float_X( r2 * PI * v * R_inv(0.0) / SPEED_OF_LIGHT );
                        float_X phase_v = phase_shift_GDD + phase_shift_TOD + phase_shift_transversal_1 + phase_shift_transversal_2 + phase_shift_transversal_3;
                        return phase_v;
                    }

                    /** fourier-transformation from frequency domain to time domain
                     * @param currentStep current simulation time step
                     * @param r2 radius to beam axis to the power of 2
                     */
                    HDINLINE float_X E_t_dft( uint32_t currentStep , float_X r2)
                    {
                        // number of steps for the fourier-transformation
                        int N = int(( int(Unitless::PULSE_INIT * Unitless::PULSE_LENGTH / DELTA_T) - 1) / 2);

	                    // currentStep as signed integer
	                    int currentStep_signed = int(currentStep);

                        // timesteps for DFT range from -N*dt to N*dt -> 2N+1 timesteps, equally spaced
                        float_X const runTime = float_X( (currentStep_signed - N) * DELTA_T);
            
                        /* Calculation of the E(t) using trigonometric Interpolation.
                        Coefficients can be easily determined cause the spectrum is given.
                        */
                        float_64 E_a = 0.0;                                                                             // for summation of symm. coeff.
                        float_64 E_b = 0.0;                                                                             // for summation of antisymm. coeff.
                        for(int k = 0; k < N+1; ++k)
                        {
                            float_X v_k = k / ((2*N + 1) * DELTA_T);                                                    // discrete values of frequency
                            float_X a = (2.0/DELTA_T) * transversal_spectrum(v_k, r2) * math::cos( phase_v(v_k, r2) );  // symm. coeff. trig. Interpolation
                            float_X b = (2.0/DELTA_T) * transversal_spectrum(v_k, r2) * math::sin( phase_v(v_k, r2) );  // antisymm. coeff. trig. Interpolation

                            if( k == 0 )
                            {
                                E_a += a / 2.0;
                            }

                            else if( k > 0 )
                            {
                                E_a += a * math::cos(2.0 * float_X(PI) * v_k * runTime);
                                E_b += b * math::sin(2.0 * float_X(PI) * v_k * runTime);
                            }

                        }

                        // E(t)-Field derived from Spectrum
                        float_64 E_t = (E_a + E_b) / float_X(2*N + 1);
                        E_t *= Unitless::AMPLITUDE;

                        return E_t;
                    }

                    /** Device-Side Constructor
                     *
                     * @param superCellToLocalOriginCellOffset local offset in cells to current supercell
                     * @param offsetToTotalDomain offset to origin of global (@todo: total) coordinate system (possibly
                     * after transform to centered origin)
                     */
                    HDINLINE PulseFromSpectrum(
                        typename FieldE::DataBoxType const& dataBoxE,
                        DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                        DataSpace<simDim> const& offsetToTotalDomain,
                        float3_X const& elong,
                        float_X const phase,
                        uint32_t const currentStep
                        )
                        : m_elong(elong)
                        , m_phase(phase)
                        , m_currentStep(currentStep)
                        , m_dataBoxE(dataBoxE)
                        , m_offsetToTotalDomain(offsetToTotalDomain)
                        , m_superCellToLocalOriginCellOffset(superCellToLocalOriginCellOffset)
                    {
                    }

                    /** device side manipulation for init plane (transversal)
                     *
                     * @tparam T_Args type of the arguments passed to the user manipulator functor
                     *
                     * @param cellIndexInSuperCell ND cell index in current supercell
                     */
                    template<typename T_Acc>
                    HDINLINE void operator()(T_Acc const&, DataSpace<simDim> const& cellIndexInSuperCell)
                    {
                        // coordinate system to global simulation as origin
                        DataSpace<simDim> const localCell(cellIndexInSuperCell + m_superCellToLocalOriginCellOffset);

                        // transform coordinate system to center of x-z plane of initialization
                        constexpr uint8_t planeNormalDir = 1u;
                        DataSpace<simDim> offsetToCenterOfPlane(m_offsetToTotalDomain);
                        offsetToCenterOfPlane[planeNormalDir] = 0; // do not shift origin of plane normal
                        floatD_X const pos
                            = precisionCast<float_X>(localCell + offsetToCenterOfPlane) * cellSize.shrink<simDim>();
                        // @todo add half-cells via traits::FieldPosition< Solver::NumicalCellType, FieldE >()

                        // transversal position only
                        floatD_X planeNoNormal = floatD_X::create(1.0_X);
                        planeNoNormal[planeNormalDir] = 0.0_X;
                        float_X const r2 = pmacc::math::abs2(pos * planeNoNormal);

                        // calculate focus position relative to the laser initialization plane
                        float_X const focusPos = Unitless::FOCUS_POS - pos.y();

                        m_elong.x() = E_t_dft(m_currentStep, r2);

                        // jump over the guard of the electric field
                        m_dataBoxE(localCell + SuperCellSize::toRT() * GuardSize::toRT()) = m_elong;

                    }
                };
            } // namespace acc

            template<typename T_Params>
            struct PulseFromSpectrum : public pulseFromSpectrum::Unitless<T_Params>
            {
                using Unitless = pulseFromSpectrum::Unitless<T_Params>;

                float3_X elong;
                float_X phase;
                uint32_t m_currentStep;
                typename FieldE::DataBoxType dataBoxE;
                DataSpace<simDim> offsetToTotalDomain;

                /** constructor
                 *
                 * @param currentStep current simulation time step
                 */
                HINLINE PulseFromSpectrum(uint32_t currentStep) : m_currentStep(currentStep)
                {
                    // get data
                    DataConnector& dc = Environment<>::get().DataConnector();
                    dataBoxE = dc.get<FieldE>(FieldE::getName(), true)->getDeviceDataBox();

                    // get meta data for offsets
                    SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                    // const DataSpace< simDim > totalCellOffset( subGrid.getGlobalDomain().offset );
                    DataSpace<simDim> const globalCellOffset(subGrid.getLocalDomain().offset);
                    DataSpace<simDim> const halfSimSize(subGrid.getGlobalDomain().size / 2);

                    // transform coordinate system to center of global simulation as origin [cells]
                    offsetToTotalDomain = /* totalCellOffset + */ globalCellOffset - halfSimSize;
                }

                /** create device manipulator functor
                 *
                 * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
                 * @tparam T_Acc alpaka accelerator type
                 *
                 * @param alpaka accelerator
                 * @param localSupercellOffset (in supercells, without guards) to the
                 *        origin of the local domain
                 * @param configuration of the worker
                 */
                template<typename T_WorkerCfg, typename T_Acc>
                HDINLINE acc::PulseFromSpectrum<Unitless> operator()(
                    T_Acc const&,
                    DataSpace<simDim> const& localSupercellOffset,
                    T_WorkerCfg const&) const
                {
                    auto const superCellToLocalOriginCellOffset = localSupercellOffset * SuperCellSize::toRT();
                    return acc::PulseFromSpectrum<Unitless>(
                        dataBoxE,
                        superCellToLocalOriginCellOffset,
                        offsetToTotalDomain,
                        elong,
                        phase,
                        m_currentStep);
                }

                //! get the name of the laser profile
                static HINLINE std::string getName()
                {
                    return "PulseFromSpectrum";
                }
            };

        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
