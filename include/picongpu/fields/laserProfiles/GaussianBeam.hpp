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
            namespace gaussianBeam
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

                    // INIT_TIME is not used in this Laser Profile!!!
                    // But Compilation without it results in following Error:
                    // LaserPhysics.hpp(109): class "picongpu:: ... ::PlaneWaveParam>"has no member "INIT_TIME"
                    // RAMP_INIT was replaced by PULSE_INIT (RAMP_INIT was defined as 1/2 PULSE_INT, but is not used further
                    // (almost the same thing happens with WAVE_LENGTH)
                    static constexpr float_X INIT_TIME = float_X( ( Params::PULSE_INIT * Params::PULSE_LENGTH_SI) / UNIT_TIME ); // unit: seconds (full inizialisation length)

                    /* initialize the laser not in the first cell is equal to a negative shift
                     * in time
                     */
                    static constexpr float_X laserTimeShift = Params::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;
                    };
            } // namespace gaussianBeam

            namespace acc
            {
                template<typename T_Unitless>
                struct GaussianBeam : public T_Unitless
                {
                    using Unitless = T_Unitless;

                    float3_X m_elong;
                    float_X m_phase;
                    float_X m_currentStep; // hier
                    typename FieldE::DataBoxType m_dataBoxE;
                    DataSpace<simDim> m_offsetToTotalDomain;
                    DataSpace<simDim> m_superCellToLocalOriginCellOffset;

                    /*
                    alpha is a factor used for the calculation of the transverse spectrum:
                    alpha = angular_frequency * waist_radius^2 / (2 * c * y)
                    y is choosen to be zero at the focus point in the frame of reference used to express the equations for the transversal spectrum.
                    as we want to initialize the laser with the focus point at FOCUS_POS in PIConGPU reference frame:
                    --> y = -Focus_POS
                    */
                    HDINLINE float_X alpha(float_X v)
                    {
                    float_X alpha = 2.0 * float_X(PI) * v * math::pow(float_X(Unitless::W0), 2.0) / (2.0 * SPEED_OF_LIGHT * -1.0 * Unitless::FOCUS_POS);
                    return alpha;
                    }

                    /*
                    gauss spectrum
                    norm is choosen so, that the maximum has value 1
                    v0 ... central frequency
                    PULSE_LENGTH ... standard sigma of gauss in time domain
                    in: v ... frequency
                    out: A_v ... amplitude depending on v (spectrum)
                    */
                    HDINLINE float_X gauss_spectrum(float_X v)
                    {
                        float_X norm = 0.5 * math::sqrt(float_X(PI)) * Unitless::PULSE_LENGTH;
                        return norm * math::exp( -1.0 * math::pow( Unitless::PULSE_LENGTH * float_X(PI) * ( v - Unitless::v0 ), 2.0) );
                    }

                    /*
                    to implement a laser-pulse with a transversal profile in this laserProfile the spectrum (and phase) have to be altered:
                    spectrum(v) --> spectrum(v, r) with radius r ( r = sqrt[ (x-x0)^2 + (z-z0)^2 ] = sqrt[x^2 + z^2] with (x0, z0) = (0, 0)
                    as you can see the the goal is to implement a radial symmetric laserProfile, which will be gaussian.
                    in addition the center axis of the beam is choosen to be at (x, z) = 0 (hense (x0, z0) = (0, 0)).

                    E_Amplitudespectrum_transversal = spectrum_v * [1 + alpha^-2]^-1/4 * exp( -r^2 / [W0^2 * [1 + alpha^-2])
                    with spectrum_v as the spectrum which one would have if a transversal profile would be neglected (here: gaussian spectrum)
                    */
                    HDINLINE float_X transversal_spectrum(float_X v, float_X r2)
                    {
                        float_X a1 = 1.0 + math::pow(alpha(v), -2.0);
                        float_X transversal_spectrum = gauss_spectrum(v) * math::pow(a1, -0.25) * math::exp( -1.0*r2 / (math::pow(float_X(Unitless::W0), 2.0) * a1 ));
                        return transversal_spectrum;
                    }

                    /*
                    This part is a bit complex, cause the phase is altered for different purposes.
                    1. to be able to choose freely the GDD/TOD of the laser-pulse:
                        --> just look at the definition of GDD/TOD as part of the Taylor-Series of the phase to understand this implementation
                    2. to implement the transversal profile:
                        -->
                    */
                    HDINLINE float_X phase_v(float_X v, float_X r2)
                    {
                        float_X phase_shift_GDD_TOD = float_X( 0.5 * Unitless::GDD * math::pow( 2.0 * float_X(PI) * ( v - Unitless::v0 ), 2.0) + (1.0/6.0) * Unitless::TOD * math::pow( 2.0 * float_X(PI) * (v - Unitless::v0), 3.0) );
                        float_X phase_shift_transversal_1 = float_X( 0.5 * math::atan(1.0/alpha(v)) );
                        float_X phase_shift_transversal_2 = float_X( 2.0 * float_X(PI) * v * Unitless::FOCUS_POS / SPEED_OF_LIGHT );
                        float_X phase_shift_transversal_3 = float_X( r2 * PI * v / ( SPEED_OF_LIGHT * Unitless::FOCUS_POS * ( 1 + math::pow(alpha(v), 2.0) ) ) );
                        float_X phase_v = phase_shift_GDD_TOD + phase_shift_transversal_1 + phase_shift_transversal_2 + phase_shift_transversal_3;
                        return phase_v;
                    }

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

                    /** Simple iteration algorithm to implement Laguerre polynomials for GPUs.
                     *
                     *  @param n order of the Laguerre polynomial
                     *  @param x coordinate at which the polynomial is evaluated
                     *  @return ...
                     
                    HDINLINE float_X simpleLaguerre(const uint32_t n, const float_X x)
                    {
                        // Result for special case n == 0
                        if(n == 0)
                            return 1.0_X;
                        uint32_t currentN = 1;
                        float_X laguerreNMinus1 = 1.0_X;
                        float_X laguerreN = 1.0_X - x;
                        float_X laguerreNPlus1(0.0_X);
                        while(currentN < n)
                        {
                            // Core statement of the algorithm
                            laguerreNPlus1 = ((2.0_X * float_X(currentN) + 1.0_X - x) * laguerreN
                                              - float_X(currentN) * laguerreNMinus1)
                                / float_X(currentN + 1u);
                            // Advance by one order
                            laguerreNMinus1 = laguerreN;
                            laguerreN = laguerreNPlus1;
                            currentN++;
                        }
                        return laguerreN;
                    }*/

                    /** Device-Side Constructor
                     *
                     * @param superCellToLocalOriginCellOffset local offset in cells to current supercell
                     * @param offsetToTotalDomain offset to origin of global (@todo: total) coordinate system (possibly
                     * after transform to centered origin)
                     */
                    HDINLINE GaussianBeam(
                        typename FieldE::DataBoxType const& dataBoxE,
                        DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                        DataSpace<simDim> const& offsetToTotalDomain,
                        float3_X const& elong,
                        float_X const phase,
                        uint32_t const currentStep
                        )
                        : m_elong(elong)
                        , m_phase(phase)
                        , m_currentStep(currentStep) //hier
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
                    HDINLINE void operator()(T_Acc const&, DataSpace<simDim> const& cellIndexInSuperCell) //hier?
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


                        /*if(Unitless::initPlaneY != 0) // compile time if
                        {
                            /* If the laser is not initialized in the first cell we emit a
                             * negatively and positively propagating wave. Therefore we need to multiply the
                             * amplitude with a correction factor depending of the cell size in
                             * propagation direction.
                             * The negatively propagating wave is damped by the absorber.
                             *
                             * The `correctionFactor` assume that the wave is moving in y direction.
                             
                            auto const correctionFactor = (SPEED_OF_LIGHT * DELTA_T) / CELL_HEIGHT * 2._X;

                            // jump over the guard of the electric field
                            m_dataBoxE(localCell + SuperCellSize::toRT() * GuardSize::toRT())
                                += correctionFactor * m_elong;
                        }
                        else
                        {
                            // jump over the guard of the electric field
                            m_dataBoxE(localCell + SuperCellSize::toRT() * GuardSize::toRT()) = m_elong;
                        }*/
                    }
                };
            } // namespace acc

            template<typename T_Params>
            struct GaussianBeam : public gaussianBeam::Unitless<T_Params>
            {
                using Unitless = gaussianBeam::Unitless<T_Params>;

                float3_X elong;
                float_X phase;
                uint32_t m_currentStep;
                typename FieldE::DataBoxType dataBoxE;
                DataSpace<simDim> offsetToTotalDomain;

                /** constructor
                 *
                 * @param currentStep current simulation time step
                 */
                HINLINE GaussianBeam(uint32_t currentStep) : m_currentStep(currentStep)
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

                    // @todo reset origin of direction of moving window
                    // offsetToTotalDomain.y() = 0

                    /*
                    float_64 const runTime = DELTA_T * currentStep - Unitless::laserTimeShift;

                    // calculate focus position relative to the laser initialization plane
                    float_X const focusPos = Unitless::FOCUS_POS - Unitless::initPlaneY * CELL_HEIGHT;

                    elong = float3_X::create(0.0_X);

                    // This check is done here on HOST, since std::numeric_limits<float_X>::epsilon() does not compile
                    // on laserTransversal(), which is on DEVICE.
                    float_X etrans_norm(0.0_X);

                    PMACC_CASSERT_MSG(
                        MODENUMBER_must_be_smaller_than_number_of_entries_in_LAGUERREMODES_vector,
                        Unitless::MODENUMBER < Unitless::LAGUERREMODES_t::dim);
                    for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                        etrans_norm += typename Unitless::LAGUERREMODES_t{}[m];
                    PMACC_VERIFY_MSG(
                        math::abs(etrans_norm) > std::numeric_limits<float_X>::epsilon(),
                        "Sum of LAGUERREMODES can not be 0.");


                    // a symmetric pulse will be initialized at position z=0 for
                    // a time of PULSE_INIT * PULSE_LENGTH = INIT_TIME.
                    // we shift the complete pulse for the half of this time to start with
                    // the front of the laser pulse.
                    constexpr float_64 mue = 0.5 * Unitless::INIT_TIME;

                    // rayleigh length (in y-direction)
                    constexpr float_64 y_R = PI * Unitless::W0 * Unitless::W0 / Unitless::WAVE_LENGTH;
                    // gaussian beam waist in the nearfield: w_y(y=0) == W0
                    float_64 const w_y = Unitless::W0 * math::sqrt(1.0 + (focusPos / y_R) * (focusPos / y_R));

                    float_64 envelope = float_64(Unitless::AMPLITUDE);
                    if(simDim == DIM2)
                        envelope *= math::sqrt(float_64(Unitless::W0) / w_y);
                    else if(simDim == DIM3)
                        envelope *= float_64(Unitless::W0) / w_y;
                    // no 1D representation/implementation

                    if(Unitless::Polarisation == Unitless::LINEAR_X)
                    {
                        elong.x() = float_X(envelope);
                    }
                    else if(Unitless::Polarisation == Unitless::LINEAR_Z)
                    {
                        elong.z() = float_X(envelope);
                    }
                    else if(Unitless::Polarisation == Unitless::CIRCULAR)
                    {
                        elong.x() = float_X(envelope) / math::sqrt(2.0_X);
                        elong.z() = float_X(envelope) / math::sqrt(2.0_X);
                    }

                    phase = 2.0_X * float_X(PI) * float_X(Unitless::f)
                            * (runTime - float_X(mue) - focusPos / SPEED_OF_LIGHT)
                        + Unitless::LASER_PHASE;
                    */
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
                HDINLINE acc::GaussianBeam<Unitless> operator()(
                    T_Acc const&,
                    DataSpace<simDim> const& localSupercellOffset,
                    T_WorkerCfg const&) const
                {
                    auto const superCellToLocalOriginCellOffset = localSupercellOffset * SuperCellSize::toRT();
                    // error: no instance of constructor "picongpu::fields::laserProfiles::acc::GaussianBeam<T_Unitless>::GaussianBeam [with T_Unitless=picongpu::fields::laserProfiles::gaussianBeam::Unitless<picongpu::fields::laserProfiles::GaussianBeamParam>]" matches the argument list
                    return acc::GaussianBeam<Unitless>(
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
                    return "GaussianBeam";
                }
            };

        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
