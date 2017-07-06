/* Copyright 2013-2017 Axel Huebl, Heiko Burau, Rene Widera, Marco Garten,
 *                     Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc_types.hpp"
#include "mappings/kernel/MappingDescription.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"
#include "fields/tasks/FieldFactory.hpp"
#include "math/Vector.hpp"
#include "traits/GetNumWorkers.hpp"

#include <boost/core/ignore_unused.hpp>


namespace PMacc
{
namespace fields
{
namespace operations
{

    /** copy guarding cells to an intermediate buffer
     *
     * @tparam T_numWorkers number of workers
     */
    template< uint32_t T_numWorkers >
    struct KernelCopyGuardToExchange
    {
        /** copy guarding cells to an intermediate box
         *
         * @tparam T_ExchangeBox PMacc::ExchangeBox, type of the intermediate box
         * @tparam T_SrcBox PMacc::DataBox, type of the local box
         * @tparam T_Extent PMacc::DataSpace, type to describe n-dimensional sizes
         * @tparam T_Mapping mapper functor type
         *
         * @param exchangeBox exchange box for the guard data of the local GPU
         * @param srcBox box to a local field
         * @param exchangeSize dimensions of exchangeBox
         * @param direction the direction of exchangeBox
         * @param mapper functor to map a CUDA block to a supercell
         */
        template<
            typename T_ExchangeBox,
            typename T_SrcBox,
            typename T_Extent,
            typename T_Mapping
        >
        DINLINE void operator()(
            T_ExchangeBox & exchangeBox,
            T_SrcBox const & srcBox,
            T_Extent const & exchangeSize,
            T_Extent const & direction,
            T_Mapping const & mapper
        ) const
        {
            using namespace mappings::threads;

            using SuperCellSize = typename T_Mapping::SuperCellSize;

            // number of cells in a superCell
            constexpr uint32_t numCells = PMacc::math::CT::volume< SuperCellSize >::type::value;
            constexpr uint32_t numWorkers = T_numWorkers;
            constexpr int dim = T_Mapping::Dim;

            uint32_t const workerIdx = threadIdx.x;

            DataSpace< dim > const blockCell(
                mapper.getSuperCellIndex( DataSpace< dim >( blockIdx ) ) *
                SuperCellSize::toRT()
            );

            // origin in area from local GPU
            DataSpace< dim > nullSourceCell(
                mapper.getSuperCellIndex( DataSpace< dim > () ) *
                SuperCellSize::toRT()
            );

            ForEachIdx<
                IdxConfig<
                    numCells,
                    numWorkers
                >
            >{ workerIdx }(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    // cell index within the superCell
                    DataSpace< dim > const cellIdx = DataSpaceOperations< dim >::template map< SuperCellSize >( linearIdx );

                    DataSpace< T_Mapping::Dim > const sourceCell( blockCell + cellIdx );
                    DataSpace< dim > targetCell( sourceCell - nullSourceCell );

                    /* defines if the virtual worker needs to copy the value of
                     * the cell to to the exchange box
                     */
                    bool copyValue = true;

                    for( uint32_t d = 0; d < dim; ++d )
                    {
                        if( direction[ d ] == -1 )
                        {
                            if( cellIdx[ d ] < SuperCellSize::toRT()[ d ] - exchangeSize[ d ] )
                                copyValue = false;
                            targetCell[ d ] -= SuperCellSize::toRT()[ d ] - exchangeSize[ d ];
                        }
                        else if( direction[d] == 1 && cellIdx[ d ] >= exchangeSize[d]  )
                            copyValue = false;
                    }

                    if( copyValue )
                        exchangeBox( targetCell ) = srcBox( sourceCell );
                }
            );
        }
    };

    /** copy guard of the local buffer to the exchange buffer
     *
     * AddExchangeToBorder is the opposite operation for the neighboring
     * device to add the exchange buffer to the local field.
     */
    struct CopyGuardToExchange
    {
        /** copy local guard to exchange buffer
         *
         * Copy data cell wise from the guard of the local to the exchange buffer.
         *
         * @tparam T_SrcBuffer PMacc::GridBuffer, type of the used buffer
         * @tparam T_SuperCellSize PMacc::math::CT::vector, size of the supercell in each direction
         *
         * @param srcBuffer source buffer with exchanges
         * @param superCellSize compile time supercell size
         * @param exchangeType the exchange direction which needs to be copied
         */
        template<
            typename T_SrcBuffer,
            typename T_SuperCellSize
        >
        void operator()(
            T_SrcBuffer & srcBuffer,
            T_SuperCellSize const & superCellSize,
            uint32_t const exchangeType
        ) const
        {
            boost::ignore_unused( superCellSize );

            using SuperCellSize = T_SuperCellSize;

            constexpr int dim = T_SuperCellSize::dim;

            using MappingDesc = MappingDescription<
                dim,
                SuperCellSize
            >;

            /* use only the x dimension to determine the number of supercells in the guard
             * PMacc restriction: all dimension must have the some number of guarding
             * supercells.
             */
            int const numGuardSuperCells = srcBuffer.getGridLayout().getGuard().x() /
                SuperCellSize::x::value;

            MappingDesc const mappingDesc(
                srcBuffer.getGridLayout().getDataSpace(),
                numGuardSuperCells,
                numGuardSuperCells
            );

            ExchangeMapping<
                GUARD,
                MappingDesc
            > mapper( mappingDesc, exchangeType );

            DataSpace< dim > const direction = Mask::getRelativeDirections< dim >(
                mapper.getExchangeType( )
            );

            constexpr uint32_t numWorkers = PMacc::traits::GetNumWorkers<
                PMacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            PMACC_KERNEL( KernelCopyGuardToExchange< numWorkers >{ } )(
                mapper.getGridDim( ),
                numWorkers
            )(
                srcBuffer.getSendExchange( exchangeType ).getDeviceBuffer( ).getDataBox( ),
                srcBuffer.getDeviceBuffer( ).getDataBox( ),
                srcBuffer.getSendExchange( exchangeType ).getDeviceBuffer( ).getDataSpace( ),
                direction,
                mapper
            );
        }
    };

} // namespace operations
} // namespace fields
} // namespace PMacc